"""
cry_gate.py — Stage 0: Cry vs Non-Cry detection gate.

FILE 2/3 for Stage 0.

Depends on: audio_features.py (File 1)

Usage:
    # Training (with pre-extracted features)
    gate = CryGate()
    metrics = gate.fit(X_train, y_train, X_val, y_val, X_test, y_test, feature_names)

    # Inference (from audio file)
    result = gate.predict(audio_path)
    if result.should_proceed:
        # pass to Stage 1, use result.soft_weight to discount predictions
"""

import pickle
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── PATH FIX: find project root so "from src.audio..." works ─────────────────
# This file lives in: babycare/src/audio/cry_gate.py
# Project root is:    babycare/
# We go up 2 levels from this file's folder to reach the project root.
_THIS_DIR = Path(__file__).resolve().parent          # .../src/audio/
_PROJECT_ROOT = _THIS_DIR.parent.parent              # .../babycare/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ──────────────────────────────────────────────────────────────────────────────

from src.audio.audio_features import (
    load_audio,
    check_quality,
    extract_cry_features,
)

EPS = 1e-12


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class GateZone(Enum):
    """The three possible outcomes of Stage 0."""
    CRY = "cry"              # P(cry) >= τ_high → confident, proceed normally
    UNCERTAIN = "uncertain"  # τ_low <= P(cry) < τ_high → proceed but discount
    NON_CRY = "non_cry"     # P(cry) < τ_low → block, don't run Stage 1


@dataclass
class GateResult:
    """What Stage 0 passes to the rest of the pipeline.

    Stage 1 should use soft_weight like this:
        P(cause_k | audio) = model_prediction * result.soft_weight
    """
    p_cry: float
    zone: GateZone
    quality_score: float
    soft_weight: float

    @property
    def should_proceed(self) -> bool:
        """Should Stage 1 run?"""
        return self.zone != GateZone.NON_CRY

    def __repr__(self):
        return (f"GateResult(p_cry={self.p_cry:.3f}, zone={self.zone.value}, "
                f"quality={self.quality_score:.2f}, sw={self.soft_weight:.3f})")


@dataclass
class GateMetrics:
    """Training evaluation results (returned by fit())."""
    roc_auc: float
    logloss: float
    brier: float
    accuracy: float
    recall_cry: float
    precision_cry: float
    f1_cry: float
    tau_low: float
    tau_high: float
    recall_at_tau_low: float
    precision_at_tau_high: float
    uncertain_fraction: float
    n_train: int
    n_val: int
    n_test: int


# ═══════════════════════════════════════════════════════════════════════════════
#  THRESHOLD LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

def _find_thresholds(
    y_true: np.ndarray,
    p_cry: np.ndarray,
    target_recall: float = 0.95,
    target_precision: float = 0.90,
) -> Tuple[float, float]:
    """Learn τ_low and τ_high from validation data.

    τ_low:  highest threshold where recall ≥ 95% (safety)
    τ_high: lowest threshold where precision ≥ 90% (confidence)
    """
    thresholds = np.linspace(0.01, 0.99, 200)

    # τ_low: sweep up, keep last threshold with good recall
    tau_low = 0.20
    for tau in thresholds:
        pred = (p_cry >= tau).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fn = np.sum((pred == 0) & (y_true == 1))
        recall = tp / (tp + fn + EPS)
        if recall >= target_recall:
            tau_low = float(tau)
        else:
            break

    # τ_high: sweep down, find first threshold with good precision
    tau_high = 0.65
    for tau in reversed(thresholds):
        pred = (p_cry >= tau).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        precision = tp / (tp + fp + EPS)
        if precision >= target_precision:
            tau_high = float(tau)
            break

    # Safety: τ_low must be below τ_high
    if tau_low >= tau_high:
        tau_low = max(0.10, tau_high - 0.25)

    return tau_low, tau_high


# ═══════════════════════════════════════════════════════════════════════════════
#  CRY GATE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class CryGate:
    """Stage 0: Probabilistic cry detection with three-zone output.

    Training: receives pre-extracted features (X, y), trains ensemble.
    Inference: loads audio file → features → classify → GateResult.
    """

    def __init__(
        self,
        sr: int = 22050,
        duration: float = 5.0,
        target_recall: float = 0.95,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.sr = sr
        self.duration = duration
        self.target_recall = target_recall
        self.rs = random_state
        self.verbose = verbose

        self.pipeline: Optional[Pipeline] = None
        self.feature_names: Optional[List[str]] = None
        self.tau_low: float = 0.25
        self.tau_high: float = 0.65
        self.is_fitted: bool = False

    # ── Training ──────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> GateMetrics:
        """Train the cry gate.

        X_train → learns the classifier
        X_val   → learns thresholds τ_low, τ_high
        X_test  → final evaluation (never touched during development)
        """
        self.feature_names = feature_names

        if self.verbose:
            n_cry_tr = np.sum(y_train == 1)
            n_non_tr = np.sum(y_train == 0)
            print(f"  Train: {len(y_train)} ({n_cry_tr} cry, {n_non_tr} non-cry)")
            print(f"  Val:   {len(y_val)}")
            print(f"  Test:  {len(y_test)}")

        # ── Build ensemble (RF + GB + LR, soft voting) ────────────────────
        rf = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=3,
            class_weight="balanced", random_state=self.rs, n_jobs=-1,
        )
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=self.rs,
        )
        lr = LogisticRegression(
            C=1.0, class_weight="balanced",
            max_iter=1000, random_state=self.rs,
        )

        ensemble = VotingClassifier(
            estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
            voting="soft", weights=[2, 1, 1],
        )

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                ensemble, method="isotonic",
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.rs),
            )),
        ])

        if self.verbose:
            print("  Training ensemble (RF + GB + LR) with calibration ...")

        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True

        # ── Learn thresholds on validation set ────────────────────────────
        p_val = self.pipeline.predict_proba(X_val)[:, 1]
        self.tau_low, self.tau_high = _find_thresholds(
            y_val, p_val, target_recall=self.target_recall,
        )

        if self.verbose:
            print(f"  Thresholds: τ_low={self.tau_low:.3f}, τ_high={self.tau_high:.3f}")

        # ── Evaluate on test set ──────────────────────────────────────────
        p_test = self.pipeline.predict_proba(X_test)[:, 1]
        metrics = self._compute_metrics(y_test, p_test, len(y_train), len(y_val))

        if self.verbose:
            print(f"\n  ── Test Results ──")
            print(f"  AUC={metrics.roc_auc:.4f}  Brier={metrics.brier:.4f}  "
                  f"LogLoss={metrics.logloss:.4f}")
            print(f"  Accuracy={metrics.accuracy:.4f}  F1={metrics.f1_cry:.4f}")
            print(f"  Recall@τ_low={metrics.recall_at_tau_low:.4f}  "
                  f"Precision@τ_high={metrics.precision_at_tau_high:.4f}")
            print(f"  Uncertain zone: {metrics.uncertain_fraction:.1%}")

        return metrics

    def _compute_metrics(self, y_true, p_cry, n_train=0, n_val=0) -> GateMetrics:
        roc = float(roc_auc_score(y_true, p_cry))
        ll = float(log_loss(y_true, np.column_stack([1 - p_cry, p_cry])))
        br = float(brier_score_loss(y_true, p_cry))

        pred_50 = (p_cry >= 0.50).astype(int)
        acc = float(accuracy_score(y_true, pred_50))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, pred_50, labels=[0, 1], zero_division=0,
        )

        pred_low = (p_cry >= self.tau_low).astype(int)
        _, rec_low, _, _ = precision_recall_fscore_support(
            y_true, pred_low, labels=[0, 1], zero_division=0,
        )

        pred_high = (p_cry >= self.tau_high).astype(int)
        prec_high, _, _, _ = precision_recall_fscore_support(
            y_true, pred_high, labels=[0, 1], zero_division=0,
        )

        uncertain = float(np.mean(
            (p_cry >= self.tau_low) & (p_cry < self.tau_high)
        ))

        return GateMetrics(
            roc_auc=roc, logloss=ll, brier=br, accuracy=acc,
            recall_cry=float(rec[1]), precision_cry=float(prec[1]),
            f1_cry=float(f1[1]),
            tau_low=self.tau_low, tau_high=self.tau_high,
            recall_at_tau_low=float(rec_low[1]),
            precision_at_tau_high=float(prec_high[1]),
            uncertain_fraction=uncertain,
            n_train=n_train, n_val=n_val, n_test=len(y_true),
        )

    # ── Prediction ────────────────────────────────────────────────────────

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        """Get P(cry) for a feature matrix. For batch evaluation."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, audio_path: str) -> GateResult:
        """Full inference for one audio file."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")

        # Step 1: Load
        y = load_audio(audio_path, sr=self.sr, duration=self.duration)
        if y is None:
            return GateResult(p_cry=0.0, zone=GateZone.NON_CRY,
                              quality_score=0.0, soft_weight=0.0)

        # Step 2: Quality check
        quality = check_quality(y, self.sr)
        q = quality["quality_score"]
        if q < 0.15:
            return GateResult(p_cry=0.0, zone=GateZone.NON_CRY,
                              quality_score=q, soft_weight=0.0)

        # Step 3: Features
        feats = extract_cry_features(y, self.sr)
        if feats is None:
            return GateResult(p_cry=0.0, zone=GateZone.NON_CRY,
                              quality_score=q, soft_weight=0.0)

        names = self.feature_names or sorted(feats.keys())
        x = np.array(
            [float(feats.get(f, 0.0)) for f in names], dtype=float
        ).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Step 4: Classify
        p_cry = float(self.pipeline.predict_proba(x)[0, 1])

        # Step 5: Three-zone decision
        if p_cry >= self.tau_high:
            zone = GateZone.CRY
            soft_weight = p_cry * q
        elif p_cry >= self.tau_low:
            zone = GateZone.UNCERTAIN
            soft_weight = p_cry * q * 0.5
        else:
            zone = GateZone.NON_CRY
            soft_weight = 0.0

        return GateResult(p_cry=p_cry, zone=zone,
                          quality_score=q, soft_weight=float(soft_weight))

    # ── Save / Load ───────────────────────────────────────────────────────

    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        if self.verbose:
            print(f"  Saved: {filepath}")

    @staticmethod
    def load(filepath: str) -> "CryGate":
        with open(filepath, "rb") as f:
            return pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    print("=" * 60)
    print("  Testing cry_gate.py with synthetic data")
    print("=" * 60)

    # Create fake data: 45 features, 500 samples
    X, y = make_classification(
        n_samples=500, n_features=45, n_informative=15,
        n_redundant=10, n_classes=2, weights=[0.4, 0.6],
        class_sep=1.5, random_state=42,
    )

    # Split: 60% train, 20% val, 20% test
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, random_state=42, stratify=y_dev,
    )

    print(f"\n  Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")

    # Train
    gate = CryGate(verbose=True)
    metrics = gate.fit(X_train, y_train, X_val, y_val, X_test, y_test,
                       feature_names=[f"feat_{i}" for i in range(45)])

    # Verify
    print(f"\n  ── Verification ──")
    assert metrics.roc_auc > 0.70
    print(f"  ✅ AUC = {metrics.roc_auc:.4f}")
    assert 0 < metrics.tau_low < metrics.tau_high < 1
    print(f"  ✅ τ_low={metrics.tau_low:.3f} < τ_high={metrics.tau_high:.3f}")

    # Test save/load
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmp = f.name
    gate.save(tmp)
    gate2 = CryGate.load(tmp)
    p1 = gate.predict_proba_batch(X_test)
    p2 = gate2.predict_proba_batch(X_test)
    assert np.allclose(p1, p2)
    print(f"  ✅ Save/Load OK")
    os.unlink(tmp)

    # Test GateResult
    r = GateResult(p_cry=0.90, zone=GateZone.CRY, quality_score=0.85, soft_weight=0.765)
    assert r.should_proceed is True
    r2 = GateResult(p_cry=0.10, zone=GateZone.NON_CRY, quality_score=0.80, soft_weight=0.0)
    assert r2.should_proceed is False
    print(f"  ✅ GateResult logic OK")

    print(f"\n{'=' * 60}")
    print(f"  cry_gate.py — ALL TESTS PASSED ✅")
    print(f"{'=' * 60}")