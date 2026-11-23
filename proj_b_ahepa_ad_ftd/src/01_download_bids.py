#!/usr/bin/env python3
"""Download OpenNeuro ds004504 participants.tsv and derivatives .set files via S3."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent.parent
BIDS_DIR = PROJ_ROOT / "data" / "ds004504"
BUCKET = "openneuro.org"
PREFIX = "ds004504"


def _make_s3():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def download_participants(s3) -> pd.DataFrame:
    BIDS_DIR.mkdir(parents=True, exist_ok=True)
    tsv_path = BIDS_DIR / "participants.tsv"
    if not tsv_path.exists():
        print("[download] Fetching participants.tsv ...")
        s3.download_file(BUCKET, f"{PREFIX}/participants.tsv", str(tsv_path))
    else:
        print(f"[cache] Using existing {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")
    print(f"[info] participants.tsv loaded: {len(df)} subjects | groups={df['Group'].value_counts().to_dict()}")
    return df


def list_derivative_sets(s3) -> dict[str, tuple[str, int]]:
    """Map participant_id -> (s3_key, size_bytes) for derivatives .set files."""
    paginator = s3.get_paginator("list_objects_v2")
    mapping: dict[str, tuple[str, int]] = {}
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{PREFIX}/derivatives/"):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("_eeg.set"):
                parts = key.split("/")
                subj = next(p for p in parts if p.startswith("sub-"))
                mapping[subj] = (key, int(item["Size"]))
    return mapping


def download_one(subj: str, key: str, expected_size: int) -> str:
    rel = key[len(PREFIX) + 1 :]  # derivatives/sub-XXX/eeg/...
    dst = BIDS_DIR / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == expected_size:
        return f"cached:{subj}"
    tmp = dst.with_suffix(".set.part")
    s3 = _make_s3()
    s3.download_file(BUCKET, key, str(tmp))
    assert tmp.stat().st_size == expected_size, f"Size mismatch for {subj}: {tmp.stat().st_size} != {expected_size}"
    tmp.replace(dst)
    return f"downloaded:{subj}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Download 12 balanced subjects (4 A, 4 C, 4 F)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    s3 = _make_s3()
    df = download_participants(s3)
    set_map = list_derivative_sets(s3)
    print(f"[info] Found {len(set_map)} derivative .set files on S3")

    if args.smoke:
        selected = []
        for grp in ["A", "C", "F"]:
            sub_ids = df[df["Group"] == grp]["participant_id"].tolist()[:4]
            selected.extend(sub_ids)
        print(f"[mode] SMOKE: downloading {len(selected)} balanced subjects: {selected}")
    else:
        selected = df["participant_id"].tolist()
        print(f"[mode] FULL: ensuring all {len(selected)} subjects")

    missing = [s for s in selected if s not in set_map]
    assert not missing, f"Missing S3 keys for subjects: {missing}"

    downloaded, cached = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(download_one, s, set_map[s][0], set_map[s][1]): s
            for s in selected
        }
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res.startswith("downloaded:"):
                downloaded += 1
            else:
                cached += 1
            if i % 10 == 0 or i == len(selected):
                print(f"[progress] {i}/{len(selected)} ready (downloaded={downloaded}, cached={cached})")

    print(f"[done] Ready subjects: {len(selected)} (downloaded={downloaded}, cached={cached})")


if __name__ == "__main__":
    main()
