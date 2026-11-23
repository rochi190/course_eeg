#!/usr/bin/env python3
"""ShallowConvNet and <50K-parameter LightGAT models for AD/FTD/CN classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowConvNet(nn.Module):
    """Schirrmeister et al. ShallowConvNet for 19-ch 250 Hz 4s EEG epochs."""

    def __init__(
        self,
        n_chans: int = 19,
        n_times: int = 1000,
        n_classes: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.temporal = nn.Conv2d(1, 40, kernel_size=(1, 25), bias=True)
        self.spatial = nn.Conv2d(40, 40, kernel_size=(n_chans, 1), bias=False)
        self.bn = nn.BatchNorm2d(40, momentum=0.1, eps=1e-5)
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.drop = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_chans, n_times)
            out = self.pool(self.bn(self.spatial(self.temporal(dummy))))
            flat_dim = out.numel()

        self.classifier = nn.Linear(flat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 19, 1000) -> (B, 1, 19, 1000)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        h = self.temporal(x)
        h = self.spatial(h)
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            h = self.bn(h.float())
            h = h ** 2
            h = self.pool(h)
            h = torch.log(torch.clamp(h, min=1e-6))
            h = self.drop(h)
            h = h.flatten(1)
            return self.classifier(h)


class EdgeAwareGATLayer(nn.Module):
    """Dense-tensor Masked Multi-Head Graph Attention incorporating edge features."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int = 10,
        num_heads: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        assert out_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.lin_node = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.lin_edge = nn.Linear(edge_dim, num_heads, bias=False)

        self.norm = nn.LayerNorm(out_dim)
        self.drop_attn = nn.Dropout(dropout * 0.5)
        self.drop_out = nn.Dropout(dropout)
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(
        self,
        h: torch.Tensor,
        edge_feat: torch.Tensor,
        adj_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        h: (B, N, in_dim)
        edge_feat: (B, N, N, edge_dim)
        adj_mask: (B, N, N) boolean mask where True indicates permitted edge
        """
        b, n, _ = h.shape
        wh = self.lin_node(h).view(b, n, self.num_heads, self.head_dim)

        # Additive attention scores: (B, H, N, N)
        src_score = torch.einsum("bnhd,hd->bhn", wh, self.a_src).unsqueeze(-1)
        dst_score = torch.einsum("bmhd,hd->bhm", wh, self.a_dst).unsqueeze(-2)
        edge_score = self.lin_edge(edge_feat).permute(0, 3, 1, 2)  # (B, H, N, N)

        e = F.leaky_relu(src_score + dst_score + edge_score, negative_slope=0.2)
        e = e.masked_fill(~adj_mask.unsqueeze(1), -1e9)
        alpha = torch.softmax(e, dim=-1)
        alpha = self.drop_attn(alpha)

        # Aggregate messages: (B, H, N, N) x (B, H, N, D) -> (B, N, H*D)
        wh_h = wh.permute(0, 2, 1, 3)
        agg = torch.matmul(alpha, wh_h).permute(0, 2, 1, 3).reshape(b, n, -1)

        out = self.norm(self.skip(h) + self.drop_out(agg))
        return F.elu(out)


class LightGAT(nn.Module):
    """Lightweight (<50K params) 2-layer Edge-Aware GAT for 19-ch EEG brain graphs."""

    def __init__(
        self,
        in_feats: int = 13,
        edge_dim: int = 10,
        hidden_dim: int = 32,
        num_heads: int = 4,
        n_nodes: int = 19,
        n_classes: int = 2,
        dropout: float = 0.5,
        topk: int = 7,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.topk = topk

        self.node_enc = nn.Sequential(
            nn.Linear(in_feats, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.pos_emb = nn.Parameter(torch.randn(1, n_nodes, hidden_dim) * 0.02)

        self.gat1 = EdgeAwareGATLayer(
            hidden_dim, hidden_dim, edge_dim=edge_dim, num_heads=num_heads, dropout=dropout
        )
        self.gat2 = EdgeAwareGATLayer(
            hidden_dim, hidden_dim, edge_dim=edge_dim, num_heads=num_heads, dropout=dropout
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, n_classes),
        )

        self.register_buffer("eye_mask", torch.eye(n_nodes, dtype=torch.bool), persistent=False)
        n_params = sum(p.numel() for p in self.parameters())
        assert n_params < 50000, f"LightGAT exceeds 50K params: {n_params}"

    def _build_topk_mask(self, wpli: torch.Tensor) -> torch.Tensor:
        """Keep top-K strongest wPLI neighbors per node plus self-loops."""
        # wpli: (B, 5, 19, 19) -> mean across bands (B, 19, 19)
        strength = wpli.mean(dim=1)
        _, top_idx = torch.topk(strength, k=min(self.topk, self.n_nodes), dim=-1)
        mask = torch.zeros_like(strength, dtype=torch.bool)
        mask.scatter_(-1, top_idx, True)
        # Symmetrize and add self-loops
        mask = mask | mask.transpose(-2, -1) | self.eye_mask.unsqueeze(0)
        return mask

    def forward(
        self,
        node_feats: torch.Tensor,
        wpli: torch.Tensor,
        plv: torch.Tensor,
    ) -> torch.Tensor:
        # node_feats: (B, 19, 13); wpli, plv: (B, 5, 19, 19)
        h = self.node_enc(node_feats) + self.pos_emb
        edge_feat = torch.cat([wpli, plv], dim=1).permute(0, 2, 3, 1)  # (B, 19, 19, 10)
        adj_mask = self._build_topk_mask(wpli)

        h = self.gat1(h, edge_feat, adj_mask)
        h = self.gat2(h, edge_feat, adj_mask)

        h_mean = h.mean(dim=1)
        h_max = h.max(dim=1).values
        g = torch.cat([h_mean, h_max], dim=-1)
        return self.head(g)
