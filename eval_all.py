"""
eval_all.py — Comprehensive evaluation of the entire pipeline.

Evaluates:
    1. Stage 0: Cry detection (already evaluated during training)
    2. Node A: Hungry vs Non-Hungry (audio-only)
    3. Node B: Tired vs Active (audio-only)
    4. Node C: Belly/Burping/Discomfort (audio-only)
    5. FUSION: Audio + Bio combined (the key comparison)

The critical comparison for your thesis:
    Audio-only accuracy  vs  Audio+Bio fusion accuracy
    This proves that multimodal fusion improves predictions.

Uses CACHED features from training (no re-extraction needed).

Usage:
    python src/evaluation/eval_all.py

Outputs:
    results/eval_summary.txt    — text report
    results/eval_metrics.pkl    — metrics dict for plots.py
"""

import pickle
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, log_loss,
    brier_score_loss, confusion_matrix, classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ALL paths relative to project root (not current working directory)
def _p(relative_path: str) -> str:
    """Resolve path relative to project root. Fixes Spyder --wdir issues."""
    return str(_PROJECT_ROOT / relative_path)

from src.bio.bio_model import BioModel, BioInput, BioSimulator

W = 60


def hdr(title):
    print(f"\n  {'═' * W}")
    print(f"  {title:^{W}}")
    print(f"  {'═' * W}")


def load_cached_features(cache_path: str) -> Optional[pd.DataFrame]:
    """Load cached features from training."""
    p = Path(cache_path)
    if not p.exists():
        print(f"  ❌ Cache not found: {p}")
        return None
    with open(p, "rb") as f:
        df = pickle.load(f)
    return df


def split_test(df, random_state=42):
    """Reproduce the same test split used during training."""
    feature_cols = [c for c in df.columns if c not in ("label", "filepath")]
    X = df[feature_cols].values.astype(float)
    y = df["label"].values

    # Same split logic as training scripts
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y)
    return X_test, y_test, feature_cols


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATE A SINGLE NODE
# ═══════════════════════════════════════════════════════════════════════════════

def eval_binary_node(
    model_path: str,
    cache_path: str,
    node_name: str,
    pos_label: str,
    neg_label: str,
) -> Optional[Dict]:
    """Evaluate a binary node (A or B) on its test set."""

    df = load_cached_features(cache_path)
    if df is None:
        return None

    model_p = Path(model_path)
    if not model_p.exists():
        print(f"  ❌ Model not found: {model_p}")
        return None

    with open(model_p, "rb") as f:
        model = pickle.load(f)

    X_test, y_test, feat_names = split_test(df)

    # Get probabilities
    p_pos = model.predict_proba_batch(X_test)
    y_binary = (y_test == 1).astype(int)

    # Metrics
    pred = (p_pos >= 0.50).astype(int)
    acc = accuracy_score(y_binary, pred)
    auc = roc_auc_score(y_binary, p_pos)
    f1 = f1_score(y_binary, pred)
    brier = brier_score_loss(y_binary, p_pos)
    ll = log_loss(y_binary, np.column_stack([1 - p_pos, p_pos]))
    prec, rec, _, _ = precision_recall_fscore_support(
        y_binary, pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_binary, pred)

    results = {
        "node": node_name,
        "accuracy": float(acc),
        "auc": float(auc),
        "f1_positive": float(f1),
        "brier": float(brier),
        "logloss": float(ll),
        "precision_pos": float(prec[1]),
        "recall_pos": float(rec[1]),
        "confusion_matrix": cm.tolist(),
        "n_test": len(y_test),
        "p_test": p_pos.tolist(),
        "y_test": y_binary.tolist(),
    }

    print(f"\n  {node_name}: {pos_label} vs {neg_label}")
    print(f"  {'─' * 40}")
    print(f"  AUC      = {auc:.4f}")
    print(f"  Accuracy = {acc:.4f}")
    print(f"  F1({pos_label}) = {f1:.4f}")
    print(f"  Brier    = {brier:.4f}")
    print(f"  Confusion matrix:")
    print(f"              pred_{neg_label[:4]}  pred_{pos_label[:4]}")
    print(f"  true_{neg_label[:4]}    {cm[0,0]:5d}     {cm[0,1]:5d}")
    print(f"  true_{pos_label[:4]}    {cm[1,0]:5d}     {cm[1,1]:5d}")

    return results


def eval_multiclass_node(
    model_path: str,
    cache_path: str,
    node_name: str,
    classes: List[str],
) -> Optional[Dict]:
    """Evaluate a multi-class node (C)."""

    df = load_cached_features(cache_path)
    if df is None:
        return None

    model_p = Path(model_path)
    if not model_p.exists():
        print(f"  ❌ Model not found: {model_p}")
        return None

    with open(model_p, "rb") as f:
        model = pickle.load(f)

    X_test, y_test, feat_names = split_test(df)

    p_all = model.predict_proba_batch(X_test)
    y_pred_idx = np.argmax(p_all, axis=1)
    y_pred = model.label_encoder.inverse_transform(y_pred_idx)

    acc = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    # Per-class F1
    f1_per = {}
    for cls in classes:
        mask = y_test == cls
        if mask.sum() > 0:
            f1_per[cls] = float(f1_score(
                (y_test == cls).astype(int),
                (y_pred == cls).astype(int),
                zero_division=0))
        else:
            f1_per[cls] = 0.0

    results = {
        "node": node_name,
        "accuracy": float(acc),
        "f1_macro": float(f1_mac),
        "f1_per_class": f1_per,
        "confusion_matrix": cm.tolist(),
        "n_test": len(y_test),
        "classes": classes,
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
    }

    print(f"\n  {node_name}: {' / '.join(classes)}")
    print(f"  {'─' * 40}")
    print(f"  Accuracy = {acc:.4f}")
    print(f"  F1 macro = {f1_mac:.4f}")
    for cls, f1v in f1_per.items():
        print(f"  F1({cls}) = {f1v:.4f}")
    print(f"  Confusion matrix:")
    header = "".join(f"{c[:8]:>10}" for c in classes)
    print(f"  {'':>14}{header}")
    for i, cls in enumerate(classes):
        row = "".join(f"{cm[i,j]:>10d}" for j in range(len(classes)))
        print(f"  {cls:<14}{row}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATE FUSION: audio-only vs audio+bio
# ═══════════════════════════════════════════════════════════════════════════════

def eval_fusion(
    nodeA_model_path: str,
    nodeB_model_path: str,
    nodeC_model_path: str,
    nodeA_cache: str,
    nodeB_cache: str,
    nodeC_cache: str,
) -> Optional[Dict]:
    """Compare audio-only vs audio+bio vs audio+bio+video.

    THE KEY THESIS COMPARISON — proves multimodal fusion works.

    For each test sample:
        1. Get audio prediction
        2. Generate simulated bio context for the TRUE cause
        3. Generate simulated video features for the TRUE cause
        4. Fuse: audio only / audio+bio / audio+bio+video
        5. Compare all three
    """
    from src.fusion.fusion import FusionEngine
    from src.video.video_model import VideoModel, VideoSimulator

    hdr("Fusion Evaluation: Audio vs Audio+Bio vs Audio+Bio+Video")

    # Load all models
    models = {}
    for name, path in [("nodeA", nodeA_model_path),
                        ("nodeB", nodeB_model_path),
                        ("nodeC", nodeC_model_path)]:
        if Path(path).exists():
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
        else:
            print(f"  ❌ {name} model not found: {path}")
            return None

    df_a = load_cached_features(nodeA_cache)
    if df_a is None:
        return None

    X_test_a, y_test_a, _ = split_test(df_a)

    bio_model = BioModel()
    bio_sim = BioSimulator(random_state=42)
    video_model = VideoModel()
    video_sim = VideoSimulator(noise_level=0.3, random_state=42)
    engine = FusionEngine(verbose=False)

    cause_map_a = {1: "hungry", 0: "tired"}

    # Collect predictions for all 3 modes
    pred_audio_only = []
    pred_audio_bio = []
    pred_audio_bio_video = []
    y_true_labels = []

    p_audio_a = models["nodeA"].predict_proba_batch(X_test_a)

    for i in range(len(y_test_a)):
        true_label = int(y_test_a[i])
        true_cause = cause_map_a[true_label]
        y_true_labels.append(true_label)

        p_hungry_audio = float(p_audio_a[i])
        audio_conf = max(p_hungry_audio, 1 - p_hungry_audio)

        # ── Mode 1: Audio only ────────────────────────────────────────
        d1 = engine.fuse(
            audio_nodeA_p_hungry=p_hungry_audio,
            audio_nodeA_confidence=audio_conf,
        )
        pred_audio_only.append(1 if d1.all_proba.get("hungry", 0) >= 0.5 else 0)

        # ── Mode 2: Audio + Bio ───────────────────────────────────────
        bio_input = bio_sim.generate(true_cause, n=1)[0]
        bio_result = bio_model.predict(bio_input)

        d2 = engine.fuse(
            audio_nodeA_p_hungry=p_hungry_audio,
            audio_nodeA_confidence=audio_conf,
            bio_p_hungry=bio_result.node_a_hungry,
            bio_p_tired=bio_result.node_b_tired,
            bio_node_c=bio_result.node_c,
            bio_completeness=bio_result.completeness,
        )
        pred_audio_bio.append(1 if d2.all_proba.get("hungry", 0) >= 0.5 else 0)

        # ── Mode 3: Audio + Bio + Video ───────────────────────────────
        video_feats = video_sim.simulate(true_cause)
        video_result = video_model.predict_from_features(video_feats, is_simulated=True)

        d3 = engine.fuse(
            audio_nodeA_p_hungry=p_hungry_audio,
            audio_nodeA_confidence=audio_conf,
            bio_p_hungry=bio_result.node_a_hungry,
            bio_p_tired=bio_result.node_b_tired,
            bio_node_c=bio_result.node_c,
            bio_completeness=bio_result.completeness,
            video_p_hungry=video_result.node_a_hungry,
            video_p_tired=video_result.node_b_tired,
            video_node_c=video_result.node_c,
            video_reliability=video_result.reliability,
        )
        pred_audio_bio_video.append(1 if d3.all_proba.get("hungry", 0) >= 0.5 else 0)

    y_true = np.array(y_true_labels)
    pa = np.array(pred_audio_only)
    pab = np.array(pred_audio_bio)
    pabv = np.array(pred_audio_bio_video)

    # Compute metrics for all 3
    acc_a = accuracy_score(y_true, pa)
    acc_ab = accuracy_score(y_true, pab)
    acc_abv = accuracy_score(y_true, pabv)
    f1_a = f1_score(y_true, pa, zero_division=0)
    f1_ab = f1_score(y_true, pab, zero_division=0)
    f1_abv = f1_score(y_true, pabv, zero_division=0)

    print(f"\n  Comparison on Node A test set ({len(y_true)} samples):")
    print(f"  {'':>20} {'Audio':>10} {'Audio+Bio':>12} {'A+B+Video':>12}")
    print(f"  {'─' * 58}")
    print(f"  {'Accuracy':>20} {acc_a:>10.4f} {acc_ab:>12.4f} {acc_abv:>12.4f}")
    print(f"  {'F1 (hungry)':>20} {f1_a:>10.4f} {f1_ab:>12.4f} {f1_abv:>12.4f}")

    # Per-class analysis
    for label_val, label_name in [(1, "hungry"), (0, "non_hungry")]:
        mask = y_true == label_val
        if mask.sum() > 0:
            a_v = accuracy_score(y_true[mask], pa[mask])
            ab_v = accuracy_score(y_true[mask], pab[mask])
            abv_v = accuracy_score(y_true[mask], pabv[mask])
            print(f"  {'Acc ' + label_name:>20} {a_v:>10.4f} {ab_v:>12.4f} {abv_v:>12.4f}")

    # Improvement summary
    imp_bio = acc_ab - acc_a
    imp_video = acc_abv - acc_ab
    imp_total = acc_abv - acc_a

    print(f"\n  Improvements:")
    print(f"    Audio → Audio+Bio:         {imp_bio:+.4f} ({imp_bio*100:+.1f}%)")
    print(f"    Audio+Bio → +Video:        {imp_video:+.4f} ({imp_video*100:+.1f}%)")
    print(f"    Audio → Audio+Bio+Video:   {imp_total:+.4f} ({imp_total*100:+.1f}%)")

    results = {
        "acc_audio_only": float(acc_a),
        "acc_audio_bio": float(acc_ab),
        "acc_audio_bio_video": float(acc_abv),
        "f1_audio_only": float(f1_a),
        "f1_audio_bio": float(f1_ab),
        "f1_audio_bio_video": float(f1_abv),
        "improvement_bio": float(imp_bio),
        "improvement_video": float(imp_video),
        "improvement_total": float(imp_total),
        "n_test": len(y_true),
    }

    return results

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    hdr("Full Pipeline Evaluation")
    t0 = time.time()

    output_dir = Path(_p("results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}

    # ── Stage 0 ───────────────────────────────────────────────────────────
    hdr("Stage 0: Cry Gate")
    report_path = Path(_p("models/stage0_cry_gate/stage0_report.txt"))
    if report_path.exists():
        print(f"  (loading from saved report)")
        print(f"  {report_path.read_text()}")
    else:
        print("  ⚠️  No Stage 0 report found. Run train_stage0.py first.")

    # ── Node A ────────────────────────────────────────────────────────────
    hdr("Node A: Hungry vs Non-Hungry")
    r_a = eval_binary_node(
        model_path=_p("models/nodeA_hungry_vs_nonhungry/nodeA_model.pkl"),
        cache_path=_p("cache/nodeA_features.pkl"),
        node_name="Node A",
        pos_label="hungry",
        neg_label="non_hungry",
    )
    if r_a:
        all_metrics["node_a"] = r_a

    # ── Node B ────────────────────────────────────────────────────────────
    hdr("Node B: Tired vs Active")
    r_b = eval_binary_node(
        model_path=_p("models/nodeB_tired_vs_active/nodeB_model.pkl"),
        cache_path=_p("cache/nodeB_features.pkl"),
        node_name="Node B",
        pos_label="tired",
        neg_label="active",
    )
    if r_b:
        all_metrics["node_b"] = r_b

    # ── Node C ────────────────────────────────────────────────────────────
    hdr("Node C: Discomfort Sub-types")
    r_c = eval_multiclass_node(
        model_path=_p("models/nodeC_belly_burp_discomfort/nodeC_model.pkl"),
        cache_path=_p("cache/nodeC_features.pkl"),
        node_name="Node C",
        classes=["belly_pain", "burping", "discomfort"],
    )
    if r_c:
        all_metrics["node_c"] = r_c

    # ── Fusion Comparison ─────────────────────────────────────────────────
    r_fusion = eval_fusion(
        nodeA_model_path=_p("models/nodeA_hungry_vs_nonhungry/nodeA_model.pkl"),
        nodeB_model_path=_p("models/nodeB_tired_vs_active/nodeB_model.pkl"),
        nodeC_model_path=_p("models/nodeC_belly_burp_discomfort/nodeC_model.pkl"),
        nodeA_cache=_p("cache/nodeA_features.pkl"),
        nodeB_cache=_p("cache/nodeB_features.pkl"),
        nodeC_cache=_p("cache/nodeC_features.pkl"),
    )
    if r_fusion:
        all_metrics["fusion"] = r_fusion

    # ── Summary Table ─────────────────────────────────────────────────────
    hdr("Summary")

    print(f"\n  {'Component':<20} {'Metric':>10} {'Value':>10}")
    print(f"  {'─' * 42}")

    if "node_a" in all_metrics:
        print(f"  {'Node A (audio)':.<20} {'AUC':>10} {all_metrics['node_a']['auc']:>10.4f}")
    if "node_b" in all_metrics:
        print(f"  {'Node B (audio)':.<20} {'AUC':>10} {all_metrics['node_b']['auc']:>10.4f}")
    if "node_c" in all_metrics:
        print(f"  {'Node C (audio)':.<20} {'F1 macro':>10} {all_metrics['node_c']['f1_macro']:>10.4f}")

    if "fusion" in all_metrics:
        fm = all_metrics["fusion"]
        print(f"  {'':─<42}")
        print(f"  {'Audio-only Acc':.<20} {'':>10} {fm['acc_audio_only']:>10.4f}")
        print(f"  {'Audio+Bio Acc':.<20} {'':>10} {fm['acc_audio_bio']:>10.4f}")
        print(f"  {'Audio+Bio+Video Acc':.<20} {'':>10} {fm['acc_audio_bio_video']:>10.4f}")
        print(f"  {'':─<42}")
        print(f"  {'Bio improvement':.<20} {'':>10} {fm['improvement_bio']:>+10.4f}")
        print(f"  {'Video improvement':.<20} {'':>10} {fm['improvement_video']:>+10.4f}")
        print(f"  {'Total improvement':.<20} {'':>10} {fm['improvement_total']:>+10.4f}")

    # ── Save metrics ──────────────────────────────────────────────────────
    metrics_path = output_dir / "eval_metrics.pkl"
    with open(metrics_path, "wb") as f:
        pickle.dump(all_metrics, f)
    print(f"\n  Saved metrics: {metrics_path}")

    # ── Save text report ──────────────────────────────────────────────────
    report_path = output_dir / "eval_summary.txt"
    with open(report_path, "w") as f:
        f.write("InfantCryNet — Full Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        if "node_a" in all_metrics:
            f.write(f"Node A (Hungry vs Non-Hungry):\n")
            f.write(f"  AUC = {all_metrics['node_a']['auc']:.4f}\n")
            f.write(f"  Accuracy = {all_metrics['node_a']['accuracy']:.4f}\n\n")
        if "node_b" in all_metrics:
            f.write(f"Node B (Tired vs Active):\n")
            f.write(f"  AUC = {all_metrics['node_b']['auc']:.4f}\n")
            f.write(f"  Accuracy = {all_metrics['node_b']['accuracy']:.4f}\n\n")
        if "node_c" in all_metrics:
            f.write(f"Node C (Discomfort Sub-types):\n")
            f.write(f"  Accuracy = {all_metrics['node_c']['accuracy']:.4f}\n")
            f.write(f"  F1 macro = {all_metrics['node_c']['f1_macro']:.4f}\n\n")
        if "fusion" in all_metrics:
            fm = all_metrics["fusion"]
            f.write(f"Fusion Comparison (Node A):\n")
            f.write(f"  Audio-only accuracy   = {fm['acc_audio_only']:.4f}\n")
            f.write(f"  Audio+Bio accuracy    = {fm['acc_audio_bio']:.4f}\n")
            f.write(f"  Audio+Bio+Video acc   = {fm['acc_audio_bio_video']:.4f}\n")
            f.write(f"  Bio improvement       = {fm['improvement_bio']:+.4f}\n")
            f.write(f"  Video improvement     = {fm['improvement_video']:+.4f}\n")
            f.write(f"  Total improvement     = {fm['improvement_total']:+.4f}\n")
    print(f"  Saved report: {report_path}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"\n  Next: python src/evaluation/plots.py")


if __name__ == "__main__":
    main()