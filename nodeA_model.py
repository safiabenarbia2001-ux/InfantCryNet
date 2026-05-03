"""
nodeA_model.py — Node A: Hungry vs Non-Hungry classifier.

FILE 2/3 for Node A.

Depends on: audio_features.py (updated with cause_features)

What this file does:
    - Trains a calibrated classifier to detect hunger cries
    - Outputs P(hungry) — a probability, not a hard label
    - Includes confidence estimation (how sure are we?)
    - Three-zone output: hungry / uncertain / non_hungry

Why this is separate from Stage 0:
    Stage 0 asks: "Is this a cry?" (easy, 45 features)
    Node A asks:  "Is this a HUNGER cry?" (harder, 97 features)

    A hungry cry vs a tired cry sound similar — both are cries.
    We need richer features (MFCC deltas, jitter, sub-bands) to
    tell them apart.

Usage:
    model = NodeAModel()
    metrics = model.fit(X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
    result = model.predict(audio_path)
    print(result)  # NodeAResult(p_hungry=0.82, zone=hungry, confidence=0.82)
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
    classification_report,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── PATH FIX ─────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ──────────────────────────────────────────────────────────────────────────────

from src.audio.audio_features import (
    load_audio,
    check_quality,
    extract_cause_features,
)

EPS = 1e-12


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class NodeAZone(Enum):
    """Three possible outcomes."""
    HUNGRY = "hungry"            # P(hungry) >= τ_high → confident hungry
    UNCERTAIN = "uncertain"      # τ_low <= P < τ_high → not sure
    NON_HUNGRY = "non_hungry"    # P(hungry) < τ_low → confident not hungry


@dataclass
class NodeAResult:
    """What Node A outputs to the fusion stage.

    p_hungry:    calibrated probability of hunger
    zone:        HUNGRY / UNCERTAIN / NON_HUNGRY
    confidence:  max(p_hungry, 1 - p_hungry) — how sure the model is
    """
    p_hungry: float
    zone: NodeAZone
    confidence: float

    @property
    def p_non_hungry(self) -> float:
        return 1.0 - self.p_hungry

    @property
    def proba_dict(self) -> Dict[str, float]:
        """Probability distribution for fusion."""
        return {"hungry": self.p_hungry, "non_hungry": self.p_non_hungry}

    def __repr__(self):
        return (f"NodeAResult(p_hungry={self.p_hungry:.3f}, "
                f"zone={self.zone.value}, conf={self.confidence:.3f})")


@dataclass
class NodeAMetrics:
    """Training evaluation results."""
    roc_auc: float
    logloss: float
    brier: float
    accuracy: float
    recall_hungry: float
    precision_hungry: float
    f1_hungry: float
    recall_non_hungry: float
    f1_non_hungry: float
    tau_low: float
    tau_high: float
    uncertain_fraction: float
    n_train: int
    n_val: int
    n_test: int


# ═══════════════════════════════════════════════════════════════════════════════
#  THRESHOLD LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

def _find_thresholds(
    y_true: np.ndarray,
    p_pos: np.ndarray,
    target_recall: float = 0.90,
    target_precision: float = 0.85,
) -> Tuple[float, float]:
    """Learn τ_low and τ_high for three-zone decision.

    For Node A, we're slightly less strict than Stage 0:
        - target_recall = 0.90 (vs 0.95 for cry gate)
        - target_precision = 0.85 (vs 0.90 for cry gate)

    Because confusing hungry with tired is less dangerous than
    missing a cry entirely. Both lead to "check on the baby",
    just with different suggestions.
    """
    thresholds = np.linspace(0.01, 0.99, 200)

    # τ_low: highest threshold with recall ≥ target
    tau_low = 0.30
    for tau in thresholds:
        pred = (p_pos >= tau).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fn = np.sum((pred == 0) & (y_true == 1))
        recall = tp / (tp + fn + EPS)
        if recall >= target_recall:
            tau_low = float(tau)
        else:
            break

    # τ_high: lowest threshold with precision ≥ target
    tau_high = 0.65
    for tau in reversed(thresholds):
        pred = (p_pos >= tau).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        precision = tp / (tp + fp + EPS)
        if precision >= target_precision:
            tau_high = float(tau)
            break

    if tau_low >= tau_high:
        tau_low = max(0.20, tau_high - 0.20)

    return tau_low, tau_high


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE A MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class NodeAModel:
    """Node A: Hungry vs Non-Hungry binary classifier.

    Same architecture as CryGate (ensemble + calibration + 3 zones)
    but with richer features and slightly different thresholds.
    """

    def __init__(
        self,
        sr: int = 22050,
        duration: float = 5.0,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.sr = sr
        self.duration = duration
        self.rs = random_state
        self.verbose = verbose

        self.pipeline: Optional[Pipeline] = None
        self.feature_names: Optional[List[str]] = None
        self.tau_low: float = 0.30
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
    ) -> NodeAMetrics:
        """Train Node A on pre-extracted cause features.

        y values: 1 = hungry, 0 = non_hungry
        """
        self.feature_names = feature_names

        if self.verbose:
            n_h = np.sum(y_train == 1)
            n_nh = np.sum(y_train == 0)
            print(f"  Train: {len(y_train)} ({n_h} hungry, {n_nh} non-hungry)")
            print(f"  Val:   {len(y_val)}")
            print(f"  Test:  {len(y_test)}")

        # ── Ensemble: same recipe as CryGate ──────────────────────────────
        rf = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3,
            class_weight="balanced", random_state=self.rs, n_jobs=-1,
        )
        gb = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1,
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
            print("  Training Node A ensemble (RF + GB + LR) ...")

        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True

        # ── Learn thresholds on validation set ────────────────────────────
        p_val = self.pipeline.predict_proba(X_val)[:, 1]
        self.tau_low, self.tau_high = _find_thresholds(y_val, p_val)

        if self.verbose:
            print(f"  Thresholds: τ_low={self.tau_low:.3f}, τ_high={self.tau_high:.3f}")

        # ── Evaluate on test set ──────────────────────────────────────────
        p_test = self.pipeline.predict_proba(X_test)[:, 1]
        metrics = self._compute_metrics(y_test, p_test, len(y_train), len(y_val))

        if self.verbose:
            print(f"\n  ── Test Results ──")
            print(f"  AUC={metrics.roc_auc:.4f}  Brier={metrics.brier:.4f}  "
                  f"LogLoss={metrics.logloss:.4f}")
            print(f"  Accuracy={metrics.accuracy:.4f}")
            print(f"  Hungry:     recall={metrics.recall_hungry:.4f}  "
                  f"precision={metrics.precision_hungry:.4f}  F1={metrics.f1_hungry:.4f}")
            print(f"  Non-hungry: recall={metrics.recall_non_hungry:.4f}  "
                  f"F1={metrics.f1_non_hungry:.4f}")
            print(f"  Uncertain zone: {metrics.uncertain_fraction:.1%}")

        return metrics

    def _compute_metrics(self, y_true, p_pos, n_train=0, n_val=0) -> NodeAMetrics:
        roc = float(roc_auc_score(y_true, p_pos))
        ll = float(log_loss(y_true, np.column_stack([1 - p_pos, p_pos])))
        br = float(brier_score_loss(y_true, p_pos))

        pred = (p_pos >= 0.50).astype(int)
        acc = float(accuracy_score(y_true, pred))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, pred, labels=[0, 1], zero_division=0,
        )

        uncertain = float(np.mean(
            (p_pos >= self.tau_low) & (p_pos < self.tau_high)
        ))

        return NodeAMetrics(
            roc_auc=roc, logloss=ll, brier=br, accuracy=acc,
            recall_hungry=float(rec[1]), precision_hungry=float(prec[1]),
            f1_hungry=float(f1[1]),
            recall_non_hungry=float(rec[0]), f1_non_hungry=float(f1[0]),
            tau_low=self.tau_low, tau_high=self.tau_high,
            uncertain_fraction=uncertain,
            n_train=n_train, n_val=n_val, n_test=len(y_true),
        )

    # ── Prediction ────────────────────────────────────────────────────────

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        """Get P(hungry) for a feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, audio_path: str) -> NodeAResult:
        """Full inference for one audio file."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")

        y = load_audio(audio_path, sr=self.sr, duration=self.duration)
        if y is None:
            return NodeAResult(p_hungry=0.5, zone=NodeAZone.UNCERTAIN, confidence=0.5)

        feats = extract_cause_features(y, self.sr)
        if feats is None:
            return NodeAResult(p_hungry=0.5, zone=NodeAZone.UNCERTAIN, confidence=0.5)

        names = self.feature_names or sorted(feats.keys())
        x = np.array(
            [float(feats.get(f, 0.0)) for f in names], dtype=float
        ).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        p_hungry = float(self.pipeline.predict_proba(x)[0, 1])
        confidence = max(p_hungry, 1.0 - p_hungry)

        if p_hungry >= self.tau_high:
            zone = NodeAZone.HUNGRY
        elif p_hungry >= self.tau_low:
            zone = NodeAZone.UNCERTAIN
        else:
            zone = NodeAZone.NON_HUNGRY

        return NodeAResult(p_hungry=p_hungry, zone=zone, confidence=confidence)

    # ── Save / Load ───────────────────────────────────────────────────────

    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        if self.verbose:
            print(f"  Saved: {filepath}")

    @staticmethod
    def load(filepath: str) -> "NodeAModel":
        with open(filepath, "rb") as f:
            return pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    print("=" * 60)
    print("  Testing nodeA_model.py with synthetic data")
    print("=" * 60)

    # 97 features (same count as cause features), 400 samples
    # Lower class_sep = harder task (hunger vs non-hunger is subtle)
    X, y = make_classification(
        n_samples=400, n_features=97, n_informative=20,
        n_redundant=15, n_classes=2, weights=[0.55, 0.45],
        class_sep=1.0, random_state=42,
    )

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, random_state=42, stratify=y_dev,
    )

    print(f"\n  Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")

    # Train
    model = NodeAModel(verbose=True)
    metrics = model.fit(X_train, y_train, X_val, y_val, X_test, y_test,
                        feature_names=[f"feat_{i}" for i in range(97)])

    # Verify
    print(f"\n  ── Verification ──")
    assert metrics.roc_auc > 0.60
    print(f"  ✅ AUC = {metrics.roc_auc:.4f}")
    assert 0 < metrics.tau_low < metrics.tau_high < 1
    print(f"  ✅ τ_low={metrics.tau_low:.3f} < τ_high={metrics.tau_high:.3f}")

    # Test save/load
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmp = f.name
    model.save(tmp)
    model2 = NodeAModel.load(tmp)
    p1 = model.predict_proba_batch(X_test)
    p2 = model2.predict_proba_batch(X_test)
    assert np.allclose(p1, p2)
    print(f"  ✅ Save/Load OK")
    os.unlink(tmp)

    # Test NodeAResult
    r = NodeAResult(p_hungry=0.85, zone=NodeAZone.HUNGRY, confidence=0.85)
    assert abs(r.proba_dict["hungry"] - 0.85) < 1e-9
    assert abs(r.proba_dict["non_hungry"] - 0.15) < 1e-9
    print(f"  ✅ NodeAResult: {r}")
    print(f"     proba_dict: {r.proba_dict}")

    print(f"\n{'=' * 60}")
    print(f"  nodeA_model.py — ALL TESTS PASSED ✅")
    print(f"{'=' * 60}")