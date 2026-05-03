"""
train_nodeB.py — Train Node B: Tired vs Active.

Data expected at:
    data/interim/nodeB/tired/    ← tired cry audio files
    data/interim/nodeB/active/   ← all other cry audio files

Usage:
    python train_nodeB.py
"""

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import List, Set

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.audio.audio_features import load_audio, extract_cause_features
from src.audio.nodeB_model import NodeBModel, NodeBZone

AUDIO_EXTENSIONS: Set[str] = {".wav", ".ogg", ".mp3", ".m4a", ".flac"}


def discover_files(folder: Path) -> List[Path]:
    if not folder.exists():
        print(f"  ❌ Folder not found: {folder}")
        return []
    return sorted([
        f for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    ])


def extract_all_features(pos_files, neg_files, sr=22050, duration=5.0, cache_path=None):
    if cache_path and Path(cache_path).exists():
        print(f"  📦 Loading cached features: {cache_path}")
        with open(cache_path, "rb") as f:
            df = pickle.load(f)
        print(f"  Loaded {len(df)} samples")
        return df

    print(f"  Extracting cause features (97) from "
          f"{len(pos_files)} tired + {len(neg_files)} active ...")

    rows = []
    n_fail = 0

    for label_val, files, tag in [(1, pos_files, "TIRED"), (0, neg_files, "ACTIVE")]:
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

    n_t = (df["label"] == 1).sum()
    n_a = (df["label"] == 0).sum()
    print(f"\n  Total: {len(df)} samples ({n_t} tired, {n_a} active)")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Failed: {n_fail}")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(df, f)
        print(f"  💾 Cached: {cache_path}")

    return df


def split_data(df, random_state=42):
    feature_cols = [c for c in df.columns if c not in ("label", "filepath")]
    X = df[feature_cols].values.astype(float)
    y = df["label"].values.astype(int)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, random_state=random_state, stratify=y_dev)
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def train_and_save(df, model_dir, random_state=42):
    X_tr, y_tr, X_val, y_val, X_te, y_te, feature_names = split_data(df, random_state)

    print(f"\n  Split:")
    print(f"    Train: {len(y_tr):4d}  (tired={np.sum(y_tr==1)}, active={np.sum(y_tr==0)})")
    print(f"    Val:   {len(y_val):4d}  (tired={np.sum(y_val==1)}, active={np.sum(y_val==0)})")
    print(f"    Test:  {len(y_te):4d}  (tired={np.sum(y_te==1)}, active={np.sum(y_te==0)})")

    model = NodeBModel(verbose=True, random_state=random_state)
    metrics = model.fit(X_tr, y_tr, X_val, y_val, X_te, y_te,
                        feature_names=feature_names)

    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out / "nodeB_model.pkl"))

    report_path = out / "nodeB_report.txt"
    with open(report_path, "w") as f:
        f.write("Node B — Tired vs Active Report\n")
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
        f.write(f"  Tired:  recall={metrics.recall_tired:.4f}  "
                f"precision={metrics.precision_tired:.4f}  "
                f"F1={metrics.f1_tired:.4f}\n")
        f.write(f"  Active: recall={metrics.recall_active:.4f}  "
                f"F1={metrics.f1_active:.4f}\n\n")
        f.write(f"  Uncertain zone: {metrics.uncertain_fraction:.1%}\n")
    print(f"  Report: {report_path}")

    return model, metrics


def quick_test(model, tired_files, active_files):
    print(f"\n  ── Quick Test (3 tired + 3 active) ──")
    rng = np.random.RandomState(42)
    test_t = rng.choice(tired_files, size=min(3, len(tired_files)), replace=False)
    test_a = rng.choice(active_files, size=min(3, len(active_files)), replace=False)

    for fp in test_t:
        r = model.predict(str(fp))
        ok = "✅" if r.zone != NodeBZone.ACTIVE else "❌ MISSED"
        print(f"  [TIRED]   P={r.p_tired:.3f}  {r.zone.value:<12}  "
              f"conf={r.confidence:.3f}  {ok}  {Path(fp).name}")

    for fp in test_a:
        r = model.predict(str(fp))
        ok = "✅" if r.zone != NodeBZone.TIRED else "⚠️ WRONG"
        print(f"  [ACTIVE]  P={r.p_tired:.3f}  {r.zone.value:<12}  "
              f"conf={r.confidence:.3f}  {ok}  {Path(fp).name}")


def main():
    parser = argparse.ArgumentParser(description="Train Node B: Tired vs Active")
    parser.add_argument(
        "--tired-dir",
        default=r"C:\Users\hp\Desktop\babycare\data\interim\nodeB\tired",
    )
    parser.add_argument(
        "--active-dir",
        default=r"C:\Users\hp\Desktop\babycare\data\interim\nodeB\active",
    )
    parser.add_argument("--model-dir", default="models/nodeB_tired_vs_active")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    W = 60
    print(f"\n  {'═' * W}")
    print(f"  {'Node B — Tired vs Active Training':^{W}}")
    print(f"  {'═' * W}")
    t0 = time.time()

    print(f"\n  Tired dir:  {args.tired_dir}")
    print(f"  Active dir: {args.active_dir}")

    tired_files = discover_files(Path(args.tired_dir))
    active_files = discover_files(Path(args.active_dir))

    if not tired_files:
        print(f"\n  ❌ No tired files found in: {args.tired_dir}")
        return
    if not active_files:
        print(f"\n  ❌ No active files found in: {args.active_dir}")
        return

    if args.limit > 0:
        tired_files = tired_files[:args.limit]
        active_files = active_files[:args.limit]

    print(f"\n  Found: {len(tired_files)} tired, {len(active_files)} active")

    print(f"\n  {'─' * W}")
    cache_path = None
    if not args.no_cache:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        cache_path = str(Path(args.cache_dir) / "nodeB_features.pkl")

    df = extract_all_features(tired_files, active_files, cache_path=cache_path)

    print(f"\n  {'─' * W}")
    model, metrics = train_and_save(df, model_dir=args.model_dir)

    print(f"\n  {'─' * W}")
    quick_test(model, tired_files, active_files)

    elapsed = time.time() - t0
    print(f"\n  {'═' * W}")
    print(f"  DONE in {elapsed:.1f}s")
    print(f"  {'═' * W}")
    print(f"\n  Model:  {args.model_dir}/nodeB_model.pkl")
    print(f"  Report: {args.model_dir}/nodeB_report.txt")
    print(f"\n  AUC            = {metrics.roc_auc:.4f}")
    print(f"  Accuracy       = {metrics.accuracy:.4f}")
    print(f"  F1 tired       = {metrics.f1_tired:.4f}")
    print(f"  Uncertain zone = {metrics.uncertain_fraction:.1%}")
    print(f"  τ_low={metrics.tau_low:.3f}  τ_high={metrics.tau_high:.3f}")
    print()


if __name__ == "__main__":
    main()
