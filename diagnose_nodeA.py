"""
diagnose_nodeA.py — Understand WHY Node A has low AUC.

Run this BEFORE trying to fix the model. It answers:
    1. Are any features different between hungry and non-hungry?
    2. How much overlap is there between classes?
    3. Is a simple model better than the ensemble?
    4. Are there subgroups inside non_hungry that confuse things?

Usage:
    python diagnose_nodeA.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── PATH FIX ─────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ──────────────────────────────────────────────────────────────────────────────

W = 60


def hdr(title):
    print(f"\n  {'═' * W}")
    print(f"  {title:^{W}}")
    print(f"  {'═' * W}")


def main():
    hdr("Node A Diagnostic Report")

    # ── Load cached features ──────────────────────────────────────────────
    cache_path = Path("cache/nodeA_features.pkl")
    if not cache_path.exists():
        print(f"  ❌ Cache not found: {cache_path}")
        print(f"  Run train_nodeA.py first to extract features.")
        return

    with open(cache_path, "rb") as f:
        df = pickle.load(f)

    feature_cols = [c for c in df.columns if c not in ("label", "filepath")]
    X = df[feature_cols].values.astype(float)
    y = df["label"].values.astype(int)

    n_hungry = np.sum(y == 1)
    n_non = np.sum(y == 0)
    print(f"\n  Dataset: {len(df)} samples ({n_hungry} hungry, {n_non} non-hungry)")
    print(f"  Features: {len(feature_cols)}")

    # ══════════════════════════════════════════════════════════════════════
    #  TEST 1: Which features differ between classes?
    # ══════════════════════════════════════════════════════════════════════
    hdr("Test 1: Feature discrimination")
    print("  Testing each feature: is it different between hungry vs non-hungry?")
    print("  Using Mann-Whitney U test (non-parametric)\n")

    X_hungry = X[y == 1]
    X_non = X[y == 0]

    results = []
    for i, feat_name in enumerate(feature_cols):
        stat, pval = stats.mannwhitneyu(
            X_hungry[:, i], X_non[:, i], alternative='two-sided'
        )
        # Effect size: rank-biserial correlation
        n1, n2 = len(X_hungry), len(X_non)
        effect_size = 1 - (2 * stat) / (n1 * n2)

        results.append({
            "feature": feat_name,
            "p_value": pval,
            "effect_size": abs(effect_size),
            "mean_hungry": np.mean(X_hungry[:, i]),
            "mean_non": np.mean(X_non[:, i]),
        })

    results_df = pd.DataFrame(results).sort_values("p_value")

    # Count significant features
    sig_001 = (results_df["p_value"] < 0.01).sum()
    sig_005 = (results_df["p_value"] < 0.05).sum()
    sig_bonf = (results_df["p_value"] < 0.05 / len(feature_cols)).sum()

    print(f"  Significant features:")
    print(f"    p < 0.05:                {sig_005:3d} / {len(feature_cols)}")
    print(f"    p < 0.01:                {sig_001:3d} / {len(feature_cols)}")
    print(f"    p < Bonferroni corrected: {sig_bonf:3d} / {len(feature_cols)}")

    # Top 10 most discriminative
    print(f"\n  Top 15 most discriminative features:")
    print(f"  {'Feature':<30} {'p-value':>10} {'Effect':>8} {'Mean H':>10} {'Mean NH':>10}")
    print(f"  {'─' * 70}")
    for _, row in results_df.head(15).iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        print(f"  {row['feature']:<30} {row['p_value']:>10.6f} {row['effect_size']:>8.4f} "
              f"{row['mean_hungry']:>10.4f} {row['mean_non']:>10.4f}  {sig}")

    # Worst features (no discrimination at all)
    print(f"\n  Bottom 5 (zero discrimination):")
    for _, row in results_df.tail(5).iterrows():
        print(f"  {row['feature']:<30} p={row['p_value']:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    #  TEST 2: Simple models — is the ensemble the problem?
    # ══════════════════════════════════════════════════════════════════════
    hdr("Test 2: Simple model comparison")
    print("  Testing if a simpler model does better than the ensemble.\n")

    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                                       max_iter=1000, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                           random_state=42, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                               random_state=42)),
        ]),
    }

    print(f"  {'Model':<25} {'AUC (5-fold CV)':>20}")
    print(f"  {'─' * 47}")
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        print(f"  {name:<25} {scores.mean():.4f} ± {scores.std():.4f}")

    # ══════════════════════════════════════════════════════════════════════
    #  TEST 3: Use ONLY the most discriminative features
    # ══════════════════════════════════════════════════════════════════════
    hdr("Test 3: Using only top features")
    print("  What if we use only the features that actually differ?\n")

    for n_top in [5, 10, 20]:
        top_feats = results_df.head(n_top)["feature"].tolist()
        top_indices = [feature_cols.index(f) for f in top_feats]
        X_top = X[:, top_indices]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                           random_state=42, n_jobs=-1)),
        ])
        scores = cross_val_score(model, X_top, y, cv=cv, scoring="roc_auc")
        print(f"  Top {n_top:2d} features: AUC = {scores.mean():.4f} ± {scores.std():.4f}")

    # ══════════════════════════════════════════════════════════════════════
    #  TEST 4: Feature overlap visualization (text-based)
    # ══════════════════════════════════════════════════════════════════════
    hdr("Test 4: Class overlap on best features")

    best_feats = results_df.head(5)["feature"].tolist()
    for feat_name in best_feats:
        idx = feature_cols.index(feat_name)
        h_vals = X_hungry[:, idx]
        n_vals = X_non[:, idx]

        h_lo, h_hi = np.percentile(h_vals, [25, 75])
        n_lo, n_hi = np.percentile(n_vals, [25, 75])

        # Overlap: how much do the interquartile ranges overlap?
        overlap_lo = max(h_lo, n_lo)
        overlap_hi = min(h_hi, n_hi)
        overlap = max(0, overlap_hi - overlap_lo)
        total_range = max(h_hi, n_hi) - min(h_lo, n_lo)
        overlap_pct = overlap / (total_range + 1e-12) * 100

        print(f"\n  {feat_name}:")
        print(f"    Hungry:      median={np.median(h_vals):.4f}  IQR=[{h_lo:.4f}, {h_hi:.4f}]")
        print(f"    Non-hungry:  median={np.median(n_vals):.4f}  IQR=[{n_lo:.4f}, {n_hi:.4f}]")
        print(f"    IQR overlap: {overlap_pct:.1f}%  {'← HIGH OVERLAP' if overlap_pct > 50 else '← some separation'}")

    # ══════════════════════════════════════════════════════════════════════
    #  VERDICT
    # ══════════════════════════════════════════════════════════════════════
    hdr("Verdict")

    best_auc_feat = results_df.iloc[0]
    if sig_005 < 5:
        print("  🔴 VERY FEW discriminative features found.")
        print("     Audio features alone may NOT separate hungry from non-hungry.")
        print("     This is expected — infant cry cause classification needs")
        print("     multimodal data (bio signals, context, video).")
        print("     → Continue building nodes B and C, then FUSION will help.")
    elif sig_005 < 20:
        print("  🟡 SOME discriminative features exist, but separation is weak.")
        print("     The model might improve with feature engineering or")
        print("     different audio representations (e.g., spectrograms + CNN).")
    else:
        print("  🟢 Many discriminative features found.")
        print("     The model should work — check for bugs or data issues.")

    print(f"\n  Most discriminative feature: {best_auc_feat['feature']} "
          f"(p={best_auc_feat['p_value']:.6f})")
    print()


if __name__ == "__main__":
    main()
