#!/usr/bin/env python3
"""1D denoising models (Bandpass, 1D-ResCNN, 1D-ResUNet) and MR-STFT loss."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt
import torch
import torch.nn as nn
import torch.nn.functional as F


_BP_SOS = butter(4, [1.0, 40.0], btype="bandpass", fs=512, output="sos")


def bandpass_1_40hz(y_batch: np.ndarray) -> np.ndarray:
    """Exp A0 baseline: 4th-order zero-phase Butterworth bandpass (1-40 Hz)."""
    out = sosfiltfilt(_BP_SOS, y_batch.astype(np.float64), axis=-1)
    return out.astype(np.float32)


class MultiScaleResBlock1D(nn.Module):
    """Parallel multi-scale (k=3,5,7) 1D residual block preserving temporal length."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.05) -> None:
        super().__init__()
        branch_ch = max(16, out_ch // 3)
        self.b3 = nn.Conv1d(in_ch, branch_ch, kernel_size=3, padding=1)
        self.b5 = nn.Conv1d(in_ch, branch_ch, kernel_size=5, padding=2)
        self.b7 = nn.Conv1d(in_ch, branch_ch, kernel_size=7, padding=3)
        self.proj = nn.Sequential(
            nn.Conv1d(branch_ch * 3, out_ch, kernel_size=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
        )
        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Conv1d(in_ch, out_ch, kernel_size=1)
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.b3(x), self.b5(x), self.b7(x)], dim=1)
        return self.act(self.proj(h) + self.skip(x))


class ResCNN1D(nn.Module):
    """Exp A1: 5-layer multi-scale 1D Residual CNN preserving T=1024."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            MultiScaleResBlock1D(64, 64),
            MultiScaleResBlock1D(64, 128),
            MultiScaleResBlock1D(128, 128),
            MultiScaleResBlock1D(128, 64),
            MultiScaleResBlock1D(64, 64),
        )
        self.head = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.blocks(h)
        return self.head(h)


class ResConvBlock1D(nn.Module):
    """UNet residual conv block using GroupNorm (stateless wrt batch variance)."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2),
            nn.GroupNorm(8, out_ch),
        )
        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Conv1d(in_ch, out_ch, kernel_size=1)
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + self.skip(x))


class BottleneckAttentionBlock(nn.Module):
    """Pre-LN Multi-Head Self-Attention + FFN block for 1D bottleneck."""

    def __init__(self, dim: int = 256, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) -> (B, L, C)
        z = x.transpose(1, 2)
        zn = self.norm1(z)
        attn_out, _ = self.attn(zn, zn, zn, need_weights=False)
        z = z + attn_out
        z = z + self.ffn(self.norm2(z))
        return z.transpose(1, 2)


class ResUNet1D(nn.Module):
    """Exp A2/A3: 4-level 1D ResUNet (32->64->128->256) with bottleneck MHSA."""

    def __init__(self, base_ch: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        # Encoder
        self.enc1 = ResConvBlock1D(1, c1, dropout)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = ResConvBlock1D(c1, c2, dropout)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = ResConvBlock1D(c2, c3, dropout)
        self.pool3 = nn.MaxPool1d(2)
        self.enc4 = ResConvBlock1D(c3, c4, dropout)
        self.pool4 = nn.MaxPool1d(2)

        # Bottleneck (L=64, C=256)
        self.bottleneck = nn.Sequential(
            ResConvBlock1D(c4, c4, dropout),
            BottleneckAttentionBlock(c4, num_heads=4, dropout=dropout),
            BottleneckAttentionBlock(c4, num_heads=4, dropout=dropout),
        )

        # Decoder
        self.up4 = nn.ConvTranspose1d(c4, c4, kernel_size=2, stride=2)
        self.dec4 = ResConvBlock1D(c4 + c4, c4, dropout)
        self.up3 = nn.ConvTranspose1d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ResConvBlock1D(c3 + c3, c3, dropout)
        self.up2 = nn.ConvTranspose1d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ResConvBlock1D(c2 + c2, c2, dropout)
        self.up1 = nn.ConvTranspose1d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ResConvBlock1D(c1 + c1, c1, dropout)

        self.head = nn.Conv1d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss (n_fft=64, 128, 256) computed safely in fp32."""

    def __init__(
        self,
        fft_sizes: tuple[int, ...] = (64, 128, 256),
        hop_sizes: tuple[int, ...] = (16, 32, 64),
        win_lengths: tuple[int, ...] = (64, 128, 256),
    ) -> None:
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        for i, wl in enumerate(win_lengths):
            self.register_buffer(f"window_{i}", torch.hann_window(wl), persistent=False)

    def _stft_mag(
        self, sig: torch.Tensor, n_fft: int, hop: int, wl: int, win: torch.Tensor
    ) -> torch.Tensor:
        spec = torch.stft(
            sig,
            n_fft=n_fft,
            hop_length=hop,
            win_length=wl,
            window=win,
            return_complex=True,
        )
        return torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-7)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Force fp32 to keep complex STFT and gradients numerically exact
        p = pred.squeeze(1).float()
        t = target.squeeze(1).float()
        total = p.new_zeros(())
        for i, (n_fft, hop, wl) in enumerate(
            zip(self.fft_sizes, self.hop_sizes, self.win_lengths)
        ):
            win = getattr(self, f"window_{i}").to(p.device, dtype=torch.float32)
            mag_p = self._stft_mag(p, n_fft, hop, wl, win)
            mag_t = self._stft_mag(t, n_fft, hop, wl, win)
            sc_loss = torch.linalg.norm(mag_t - mag_p, dim=(-2, -1)) / (
                torch.linalg.norm(mag_t, dim=(-2, -1)) + 1e-6
            )
            log_loss = F.l1_loss(
                torch.log(mag_p + 1e-5), torch.log(mag_t + 1e-5)
            )
            total = total + sc_loss.mean() + log_loss
        return total / len(self.fft_sizes)


class CombinedDenoiseLoss(nn.Module):
    """Time-domain MSE + alpha * Multi-Resolution STFT Loss."""

    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.mr_stft = MultiResolutionSTFTLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_t = self.mse(pred.float(), target.float())
        loss_f = self.mr_stft(pred, target)
        return loss_t + self.alpha * loss_f
