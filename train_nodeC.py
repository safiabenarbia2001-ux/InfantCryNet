"""
train_nodeC.py — Train Node C: Belly Pain / Burping / Discomfort.

Data expected at:
    data/interim/nodec/belly_pain/
    data/interim/nodec/burping/
    data/interim/nodec/discomfort/

Usage:
    python train_nodeC.py
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
from src.audio.nodeC_model import NodeCModel, NodeCZone, NODE_C_CLASSES

AUDIO_EXTENSIONS: Set[str] = {".wav", ".ogg", ".mp3", ".m4a", ".flac"}


def discover_files(folder: Path) -> List[Path]:
    if not folder.exists():
        print(f"  ❌ Folder not found: {folder}")
        return []
    return sorted([
        f for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    ])


def extract_all_features(
    class_files: dict,
    sr: int = 22050,
    duration: float = 5.0,
    cache_path: str = None,
) -> pd.DataFrame:
    """Extract cause features for all 3 classes.

    class_files: {"belly_pain": [paths], "burping": [paths], "discomfort": [paths]}
    """
    if cache_path and Path(cache_path).exists():
        print(f"  📦 Loading cached features: {cache_path}")
        with open(cache_path, "rb") as f:
            df = pickle.load(f)
        print(f"  Loaded {len(df)} samples")
        return df

    total = sum(len(v) for v in class_files.values())
    print(f"  Extracting cause features (97) from {total} files ...")

    rows = []
    n_fail = 0

    for class_name, files in class_files.items():
        tag = class_name.upper()
        n_ok = 0
        print(f"\n  [{tag}] Processing {len(files)} files ...")
        for i, fp in enumerate(files):
            y = load_audio(str(fp), sr=sr, duration=duration)
            if y is not None:
                feats = extract_cause_features(y, sr)
                if feats is not None:
                    feats["label"] = class_name
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

    print(f"\n  Total: {len(df)} samples")
    for cls in NODE_C_CLASSES:
        n = (df["label"] == cls).sum()
        print(f"    {cls:<14} {n:4d}")
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
    y = df["label"].values  # string labels

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, random_state=random_state, stratify=y_dev)
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def train_and_save(df, model_dir, random_state=42):
    X_tr, y_tr, X_val, y_val, X_te, y_te, feature_names = split_data(df, random_state)

    print(f"\n  Split:")
    for split_name, y_split in [("Train", y_tr), ("Val", y_val), ("Test", y_te)]:
        counts = {cls: np.sum(y_split == cls) for cls in NODE_C_CLASSES}
        parts = "  ".join(f"{c}={n}" for c, n in counts.items())
        print(f"    {split_name:>5}: {len(y_split):4d}  ({parts})")

    model = NodeCModel(verbose=True, random_state=random_state)
    metrics = model.fit(X_tr, y_tr, X_val, y_val, X_te, y_te,
                        feature_names=feature_names)

    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out / "nodeC_model.pkl"))

    report_path = out / "nodeC_report.txt"
    with open(report_path, "w") as f:
        f.write("Node C — Discomfort Sub-type Report\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Classes: {NODE_C_CLASSES}\n")
        f.write(f"Train: {len(y_tr)}  Val: {len(y_val)}  Test: {len(y_te)}\n\n")
        f.write(f"Confidence threshold: {metrics.conf_threshold:.4f}\n\n")
        f.write(f"Test Metrics:\n")
        f.write(f"  Accuracy  = {metrics.accuracy:.4f}\n")
        f.write(f"  F1 macro  = {metrics.f1_macro:.4f}\n")
        f.write(f"  LogLoss   = {metrics.logloss:.4f}\n\n")
        f.write(f"Per-class F1:\n")
        for cls, f1 in metrics.f1_per_class.items():
            f.write(f"  {cls:<14} F1={f1:.4f}\n")
        f.write(f"\nMean confidence: {metrics.confidence_mean:.4f}\n")
        f.write(f"Uncertain zone:  {metrics.uncertain_fraction:.1%}\n")
    print(f"  Report: {report_path}")

    return model, metrics


def quick_test(model, class_files):
    print(f"\n  ── Quick Test (2 per class) ──")
    rng = np.random.RandomState(42)

    for cls_name, files in class_files.items():
        if len(files) == 0:
            continue
        samples = rng.choice(files, size=min(2, len(files)), replace=False)
        for fp in samples:
            r = model.predict(str(fp))
            ok = "✅" if r.prediction == cls_name else f"→ {r.prediction}"
            print(f"  [{cls_name:<14}] pred={r.prediction:<14} "
                  f"conf={r.confidence:.3f}  {r.zone.value:<10}  "
                  f"{ok}  {Path(fp).name}")


def main():
    parser = argparse.ArgumentParser(description="Train Node C: Discomfort Sub-types")
    parser.add_argument(
        "--belly-dir",
        default=r"C:\Users\hp\Desktop\babycare\data\interim\nodec\belly_pain",
    )
    parser.add_argument(
        "--burping-dir",
        default=r"C:\Users\hp\Desktop\babycare\data\interim\nodec\burping",
    )
    parser.add_argument(
        "--discomfort-dir",
        default=r"C:\Users\hp\Desktop\babycare\data\interim\nodec\discomfort",
    )
    parser.add_argument("--model-dir", default="models/nodeC_belly_burp_discomfort")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    W = 60
    print(f"\n  {'═' * W}")
    print(f"  {'Node C — Discomfort Sub-type Training':^{W}}")
    print(f"  {'═' * W}")
    t0 = time.time()

    print(f"\n  Belly pain dir:  {args.belly_dir}")
    print(f"  Burping dir:     {args.burping_dir}")
    print(f"  Discomfort dir:  {args.discomfort_dir}")

    belly_files = discover_files(Path(args.belly_dir))
    burping_files = discover_files(Path(args.burping_dir))
    discomfort_files = discover_files(Path(args.discomfort_dir))

    if not belly_files:
        print(f"\n  ❌ No belly_pain files found")
        return
    if not burping_files:
        print(f"\n  ❌ No burping files found")
        return
    if not discomfort_files:
        print(f"\n  ❌ No discomfort files found")
        return

    if args.limit > 0:
        belly_files = belly_files[:args.limit]
        burping_files = burping_files[:args.limit]
        discomfort_files = discomfort_files[:args.limit]

    print(f"\n  Found: {len(belly_files)} belly_pain, "
          f"{len(burping_files)} burping, {len(discomfort_files)} discomfort")

    class_files = {
        "belly_pain": belly_files,
        "burping": burping_files,
        "discomfort": discomfort_files,
    }

    print(f"\n  {'─' * W}")
    cache_path = None
    if not args.no_cache:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        cache_path = str(Path(args.cache_dir) / "nodeC_features.pkl")

    df = extract_all_features(class_files, cache_path=cache_path)

    print(f"\n  {'─' * W}")
    model, metrics = train_and_save(df, model_dir=args.model_dir)

    print(f"\n  {'─' * W}")
    quick_test(model, class_files)

    elapsed = time.time() - t0
    print(f"\n  {'═' * W}")
    print(f"  DONE in {elapsed:.1f}s")
    print(f"  {'═' * W}")
    print(f"\n  Model:  {args.model_dir}/nodeC_model.pkl")
    print(f"  Report: {args.model_dir}/nodeC_report.txt")
    print(f"\n  Accuracy       = {metrics.accuracy:.4f}")
    print(f"  F1 macro       = {metrics.f1_macro:.4f}")
    for cls, f1 in metrics.f1_per_class.items():
        print(f"  F1 {cls:<12} = {f1:.4f}")
    print(f"  Uncertain zone = {metrics.uncertain_fraction:.1%}")
    print()


if __name__ == "__main__":
    main()
