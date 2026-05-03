"""
train_stage0.py — Train the Stage 0 Cry Gate on real data.

FILE 3/3 for Stage 0.

Depends on:
    audio_features.py (File 1) — feature extraction
    cry_gate.py       (File 2) — classifier

Usage:
    python train_stage0.py
    python train_stage0.py --cry-dir "path/to/cry" --noncry-dir "path/to/non_cry"
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import List, Set

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── PATH FIX ─────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ──────────────────────────────────────────────────────────────────────────────

from src.audio.audio_features import load_audio, extract_cry_features
from src.audio.cry_gate import CryGate, GateZone

AUDIO_EXTENSIONS: Set[str] = {".wav", ".ogg", ".mp3", ".m4a", ".flac"}


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1: DISCOVER AUDIO FILES
# ═══════════════════════════════════════════════════════════════════════════════

def discover_files(folder: Path) -> List[Path]:
    """Find all audio files in a folder (recursive)."""
    if not folder.exists():
        print(f"  ❌ Folder not found: {folder}")
        return []
    files = sorted([
        f for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    ])
    return files


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2: EXTRACT FEATURES (with caching)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_all_features(
    cry_files: List[Path],
    noncry_files: List[Path],
    sr: int = 22050,
    duration: float = 5.0,
    cache_path: str = None,
) -> pd.DataFrame:
    """Extract features from all files, return DataFrame with label column.

    Caches to disk so the second run is instant.
    """
    # Try cache first
    if cache_path and Path(cache_path).exists():
        print(f"  📦 Loading cached features: {cache_path}")
        with open(cache_path, "rb") as f:
            df = pickle.load(f)
        print(f"  Loaded {len(df)} samples from cache")
        return df

    # Extract from scratch
    print(f"  Extracting features from {len(cry_files)} cry + "
          f"{len(noncry_files)} non-cry files ...")

    rows = []
    n_fail = 0

    for label_val, files, tag in [(1, cry_files, "CRY"), (0, noncry_files, "NON-CRY")]:
        n_ok = 0
        print(f"\n  [{tag}] Processing {len(files)} files ...")
        for i, fp in enumerate(files):
            y = load_audio(str(fp), sr=sr, duration=duration)
            if y is not None:
                feats = extract_cry_features(y, sr)
                if feats is not None:
                    feats["label"] = label_val
                    feats["filepath"] = str(fp)
                    rows.append(feats)
                    n_ok += 1
                else:
                    n_fail += 1
            else:
                n_fail += 1

            if (i + 1) % 50 == 0:
                print(f"    [{i + 1}/{len(files)}]  ok={n_ok}  fail={n_fail}")

        print(f"  [{tag}] Done: {n_ok} extracted")

    if not rows:
        raise RuntimeError("No features extracted! Check audio files and librosa.")

    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c not in ("label", "filepath")]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    n_cry = (df["label"] == 1).sum()
    n_non = (df["label"] == 0).sum()
    print(f"\n  Total: {len(df)} samples ({n_cry} cry, {n_non} non-cry)")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Failed: {n_fail}")

    # Save to cache
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(df, f)
        print(f"  💾 Cached: {cache_path}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3: SPLIT DATA
# ═══════════════════════════════════════════════════════════════════════════════

def split_data(df: pd.DataFrame, random_state: int = 42):
    """Split into train 60% / val 20% / test 20%.

    Returns (X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
    """
    feature_cols = [c for c in df.columns if c not in ("label", "filepath")]
    X = df[feature_cols].values.astype(float)
    y = df["label"].values.astype(int)

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, random_state=random_state, stratify=y_dev,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4-5: TRAIN + SAVE
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_save(df: pd.DataFrame, model_dir: str, random_state: int = 42):
    """Split, train, save model and report."""

    X_tr, y_tr, X_val, y_val, X_te, y_te, feature_names = split_data(df, random_state)

    print(f"\n  Split:")
    print(f"    Train: {len(y_tr):4d}  (cry={np.sum(y_tr==1)}, non-cry={np.sum(y_tr==0)})")
    print(f"    Val:   {len(y_val):4d}  (cry={np.sum(y_val==1)}, non-cry={np.sum(y_val==0)})")
    print(f"    Test:  {len(y_te):4d}  (cry={np.sum(y_te==1)}, non-cry={np.sum(y_te==0)})")

    # Train
    gate = CryGate(verbose=True, random_state=random_state)
    metrics = gate.fit(X_tr, y_tr, X_val, y_val, X_te, y_te,
                       feature_names=feature_names)

    # Save model
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    gate.save(str(out / "cry_gate.pkl"))

    # Save readable report
    report_path = out / "stage0_report.txt"
    with open(report_path, "w") as f:
        f.write("Stage 0 — Cry Gate Training Report\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Train: {len(y_tr)}  Val: {len(y_val)}  Test: {len(y_te)}\n\n")
        f.write(f"Thresholds:\n")
        f.write(f"  τ_low  = {metrics.tau_low:.4f}\n")
        f.write(f"  τ_high = {metrics.tau_high:.4f}\n\n")
        f.write(f"Test Metrics:\n")
        f.write(f"  AUC       = {metrics.roc_auc:.4f}\n")
        f.write(f"  Brier     = {metrics.brier:.4f}\n")
        f.write(f"  LogLoss   = {metrics.logloss:.4f}\n")
        f.write(f"  Accuracy  = {metrics.accuracy:.4f}\n")
        f.write(f"  Recall    = {metrics.recall_cry:.4f}\n")
        f.write(f"  Precision = {metrics.precision_cry:.4f}\n")
        f.write(f"  F1        = {metrics.f1_cry:.4f}\n\n")
        f.write(f"Safety:\n")
        f.write(f"  Recall@τ_low     = {metrics.recall_at_tau_low:.4f}\n")
        f.write(f"  Precision@τ_high = {metrics.precision_at_tau_high:.4f}\n")
        f.write(f"  Uncertain zone   = {metrics.uncertain_fraction:.1%}\n")
    print(f"  Report: {report_path}")

    return gate, metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 6: QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

def quick_test(gate: CryGate, cry_files: List[Path], noncry_files: List[Path]):
    """Test on a few random files to see real predictions."""
    print(f"\n  ── Quick Test (3 cry + 3 non-cry) ──")

    rng = np.random.RandomState(42)
    test_cry = rng.choice(cry_files, size=min(3, len(cry_files)), replace=False)
    test_non = rng.choice(noncry_files, size=min(3, len(noncry_files)), replace=False)

    for fp in test_cry:
        r = gate.predict(str(fp))
        ok = "✅" if r.zone != GateZone.NON_CRY else "❌ MISSED"
        print(f"  [CRY]     P={r.p_cry:.3f}  {r.zone.value:<10}  "
              f"q={r.quality_score:.2f}  sw={r.soft_weight:.3f}  "
              f"{ok}  {Path(fp).name}")

    for fp in test_non:
        r = gate.predict(str(fp))
        ok = "✅" if r.zone == GateZone.NON_CRY else "⚠️ FALSE ALARM"
        print(f"  [NON-CRY] P={r.p_cry:.3f}  {r.zone.value:<10}  "
              f"q={r.quality_score:.2f}  sw={r.soft_weight:.3f}  "
              f"{ok}  {Path(fp).name}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train Stage 0: Cry Gate")
    parser.add_argument(
        "--cry-dir",
        default=r"C:\Users\hp\Desktop\babycare\data\interim\stage_0\cry",
    )
    parser.add_argument(
        "--noncry-dir",
        default=r"C:\Users\hp\Desktop\babycare\data\interim\stage_0\non_cry",
    )
    parser.add_argument("--model-dir", default="models/stage0_cry_gate")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Max files per class (0=all)")
    args = parser.parse_args()

    W = 60
    print(f"\n  {'═' * W}")
    print(f"  {'Stage 0 — Cry Gate Training':^{W}}")
    print(f"  {'═' * W}")
    t0 = time.time()

    # Step 1: Discover
    print(f"\n  Cry dir:     {args.cry_dir}")
    print(f"  Non-cry dir: {args.noncry_dir}")

    cry_files = discover_files(Path(args.cry_dir))
    noncry_files = discover_files(Path(args.noncry_dir))

    if not cry_files:
        print(f"\n  ❌ No cry files found in: {args.cry_dir}")
        return
    if not noncry_files:
        print(f"\n  ❌ No non-cry files found in: {args.noncry_dir}")
        return

    if args.limit > 0:
        cry_files = cry_files[:args.limit]
        noncry_files = noncry_files[:args.limit]

    print(f"\n  Found: {len(cry_files)} cry, {len(noncry_files)} non-cry")

    # Step 2: Extract
    print(f"\n  {'─' * W}")
    cache_path = None
    if not args.no_cache:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        cache_path = str(Path(args.cache_dir) / "stage0_features.pkl")

    df = extract_all_features(cry_files, noncry_files, cache_path=cache_path)

    # Step 3-5: Train + Save
    print(f"\n  {'─' * W}")
    gate, metrics = train_and_save(df, model_dir=args.model_dir)

    # Step 6: Quick test
    print(f"\n  {'─' * W}")
    quick_test(gate, cry_files, noncry_files)

    # Summary
    elapsed = time.time() - t0
    print(f"\n  {'═' * W}")
    print(f"  DONE in {elapsed:.1f}s")
    print(f"  {'═' * W}")
    print(f"\n  Model:  {args.model_dir}/cry_gate.pkl")
    print(f"  Report: {args.model_dir}/stage0_report.txt")
    print(f"\n  AUC            = {metrics.roc_auc:.4f}")
    print(f"  Recall@τ_low   = {metrics.recall_at_tau_low:.4f}")
    print(f"  Uncertain zone = {metrics.uncertain_fraction:.1%}")
    print(f"  τ_low={metrics.tau_low:.3f}  τ_high={metrics.tau_high:.3f}")
    print()


if __name__ == "__main__":
    main()