"""
train_nodeA.py — Train Node A: Hungry vs Non-Hungry.

FILE 3/3 for Node A.

Depends on:
    audio_features.py  — cause_features extraction (97 features)
    nodeA_model.py     — NodeAModel classifier

Data expected at:
    data/interim/nodeA/hungry/      ← hungry cry audio files
    data/interim/nodeA/non_hungry/  ← all other cry audio files

Usage:
    python train_nodeA.py
    python train_nodeA.py --hungry-dir "path/to/hungry" --nonhungry-dir "path/to/non_hungry"
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

from src.audio.audio_features import load_audio, extract_cause_features
from src.audio.nodeA_model import NodeAModel, NodeAZone

AUDIO_EXTENSIONS: Set[str] = {".wav", ".ogg", ".mp3", ".m4a", ".flac"}


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1: DISCOVER
# ═══════════════════════════════════════════════════════════════════════════════

def discover_files(folder: Path) -> List[Path]:
    if not folder.exists():
        print(f"  ❌ Folder not found: {folder}")
        return []
    return sorted([
        f for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    ])


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2: EXTRACT CAUSE FEATURES (with caching)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_all_features(
    hungry_files: List[Path],
    nonhungry_files: List[Path],
    sr: int = 22050,
    duration: float = 5.0,
    cache_path: str = None,
) -> pd.DataFrame:
    """Extract cause features (97) from all files."""

    if cache_path and Path(cache_path).exists():
        print(f"  📦 Loading cached features: {cache_path}")
        with open(cache_path, "rb") as f:
            df = pickle.load(f)
        print(f"  Loaded {len(df)} samples")
        return df

    print(f"  Extracting cause features (97) from "
          f"{len(hungry_files)} hungry + {len(nonhungry_files)} non-hungry ...")

    rows = []
    n_fail = 0

    for label_val, files, tag in [(1, hungry_files, "HUNGRY"),
                                   (0, nonhungry_files, "NON-HUNGRY")]:
        n_ok = 0
        print(f"\n  [{tag}] Processing {len(files)} files ...")
        for i, fp in enumerate(files):
            y = load_audio(str(fp), sr=sr, duration=duration)
            if y is not None:
                feats = extract_cause_features(y, sr)
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
        raise RuntimeError("No features extracted!")

    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c not in ("label", "filepath")]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    n_h = (df["label"] == 1).sum()
    n_nh = (df["label"] == 0).sum()
    print(f"\n  Total: {len(df)} samples ({n_h} hungry, {n_nh} non-hungry)")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Failed: {n_fail}")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(df, f)
        print(f"  💾 Cached: {cache_path}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3: SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

def split_data(df: pd.DataFrame, random_state: int = 42):
    """60% train / 20% val / 20% test, stratified."""
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
    X_tr, y_tr, X_val, y_val, X_te, y_te, feature_names = split_data(df, random_state)

    print(f"\n  Split:")
    print(f"    Train: {len(y_tr):4d}  (hungry={np.sum(y_tr==1)}, "
          f"non-hungry={np.sum(y_tr==0)})")
    print(f"    Val:   {len(y_val):4d}  (hungry={np.sum(y_val==1)}, "
          f"non-hungry={np.sum(y_val==0)})")
    print(f"    Test:  {len(y_te):4d}  (hungry={np.sum(y_te==1)}, "
          f"non-hungry={np.sum(y_te==0)})")

    model = NodeAModel(verbose=True, random_state=random_state)
    metrics = model.fit(X_tr, y_tr, X_val, y_val, X_te, y_te,
                        feature_names=feature_names)

    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out / "nodeA_model.pkl"))

    # Save report
    report_path = out / "nodeA_report.txt"
    with open(report_path, "w") as f:
        f.write("Node A — Hungry vs Non-Hungry Report\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Train: {len(y_tr)}  Val: {len(y_val)}  Test: {len(y_te)}\n\n")
        f.write(f"Thresholds:\n")
        f.write(f"  τ_low  = {metrics.tau_low:.4f}\n")
        f.write(f"  τ_high = {metrics.tau_high:.4f}\n\n")
        f.write(f"Test Metrics:\n")
        f.write(f"  AUC       = {metrics.roc_auc:.4f}\n")
        f.write(f"  Brier     = {metrics.brier:.4f}\n")
        f.write(f"  LogLoss   = {metrics.logloss:.4f}\n")
        f.write(f"  Accuracy  = {metrics.accuracy:.4f}\n\n")
        f.write(f"  Hungry:     recall={metrics.recall_hungry:.4f}  "
                f"precision={metrics.precision_hungry:.4f}  "
                f"F1={metrics.f1_hungry:.4f}\n")
        f.write(f"  Non-hungry: recall={metrics.recall_non_hungry:.4f}  "
                f"F1={metrics.f1_non_hungry:.4f}\n\n")
        f.write(f"  Uncertain zone: {metrics.uncertain_fraction:.1%}\n")
    print(f"  Report: {report_path}")

    return model, metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 6: QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

def quick_test(model: NodeAModel, hungry_files: List[Path], nonhungry_files: List[Path]):
    print(f"\n  ── Quick Test (3 hungry + 3 non-hungry) ──")

    rng = np.random.RandomState(42)
    test_h = rng.choice(hungry_files, size=min(3, len(hungry_files)), replace=False)
    test_nh = rng.choice(nonhungry_files, size=min(3, len(nonhungry_files)), replace=False)

    for fp in test_h:
        r = model.predict(str(fp))
        ok = "✅" if r.zone != NodeAZone.NON_HUNGRY else "❌ MISSED"
        print(f"  [HUNGRY]      P={r.p_hungry:.3f}  {r.zone.value:<12}  "
              f"conf={r.confidence:.3f}  {ok}  {Path(fp).name}")

    for fp in test_nh:
        r = model.predict(str(fp))
        ok = "✅" if r.zone != NodeAZone.HUNGRY else "⚠️ WRONG"
        print(f"  [NON-HUNGRY]  P={r.p_hungry:.3f}  {r.zone.value:<12}  "
              f"conf={r.confidence:.3f}  {ok}  {Path(fp).name}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train Node A: Hungry vs Non-Hungry")
    parser.add_argument(
        "--hungry-dir",
        default=r"C:\Users\hp\Desktop\babycare\data\interim\nodeA\hungry",
    )
    parser.add_argument(
        "--nonhungry-dir",
        default=r"C:\Users\hp\Desktop\babycare\data\interim\nodeA\non_hungry",
    )
    parser.add_argument("--model-dir", default="models/nodeA_hungry_vs_nonhungry")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    W = 60
    print(f"\n  {'═' * W}")
    print(f"  {'Node A — Hungry vs Non-Hungry Training':^{W}}")
    print(f"  {'═' * W}")
    t0 = time.time()

    # Step 1: Discover
    print(f"\n  Hungry dir:      {args.hungry_dir}")
    print(f"  Non-hungry dir:  {args.nonhungry_dir}")

    hungry_files = discover_files(Path(args.hungry_dir))
    nonhungry_files = discover_files(Path(args.nonhungry_dir))

    if not hungry_files:
        print(f"\n  ❌ No hungry files found in: {args.hungry_dir}")
        return
    if not nonhungry_files:
        print(f"\n  ❌ No non-hungry files found in: {args.nonhungry_dir}")
        return

    if args.limit > 0:
        hungry_files = hungry_files[:args.limit]
        nonhungry_files = nonhungry_files[:args.limit]

    print(f"\n  Found: {len(hungry_files)} hungry, {len(nonhungry_files)} non-hungry")

    # Step 2: Extract
    print(f"\n  {'─' * W}")
    cache_path = None
    if not args.no_cache:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        cache_path = str(Path(args.cache_dir) / "nodeA_features.pkl")

    df = extract_all_features(hungry_files, nonhungry_files, cache_path=cache_path)

    # Step 3-5: Train + Save
    print(f"\n  {'─' * W}")
    model, metrics = train_and_save(df, model_dir=args.model_dir)

    # Step 6: Quick test
    print(f"\n  {'─' * W}")
    quick_test(model, hungry_files, nonhungry_files)

    # Summary
    elapsed = time.time() - t0
    print(f"\n  {'═' * W}")
    print(f"  DONE in {elapsed:.1f}s")
    print(f"  {'═' * W}")
    print(f"\n  Model:  {args.model_dir}/nodeA_model.pkl")
    print(f"  Report: {args.model_dir}/nodeA_report.txt")
    print(f"\n  AUC            = {metrics.roc_auc:.4f}")
    print(f"  Accuracy       = {metrics.accuracy:.4f}")
    print(f"  F1 hungry      = {metrics.f1_hungry:.4f}")
    print(f"  Uncertain zone = {metrics.uncertain_fraction:.1%}")
    print(f"  τ_low={metrics.tau_low:.3f}  τ_high={metrics.tau_high:.3f}")
    print()


if __name__ == "__main__":
    main()
