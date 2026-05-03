"""
nodeB_model.py — Node B: Tired vs Active classifier.

Depends on: audio_features.py (cause_features)

Data:
    tired  = baby is sleepy/exhausted
    active = baby is crying for another reason (not tiredness)

Same architecture as Node A. Expected to have similar difficulty
(audio alone may not be enough — fusion will help).
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

# ── PATH FIX ─────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ──────────────────────────────────────────────────────────────────────────────

from src.audio.audio_features import load_audio, extract_cause_features

EPS = 1e-12


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class NodeBZone(Enum):
    TIRED = "tired"
    UNCERTAIN = "uncertain"
    ACTIVE = "active"


@dataclass
class NodeBResult:
    """What Node B outputs to the fusion stage."""
    p_tired: float
    zone: NodeBZone
    confidence: float

    @property
    def p_active(self) -> float:
        return 1.0 - self.p_tired

    @property
    def proba_dict(self) -> Dict[str, float]:
        return {"tired": self.p_tired, "active": self.p_active}

    def __repr__(self):
        return (f"NodeBResult(p_tired={self.p_tired:.3f}, "
                f"zone={self.zone.value}, conf={self.confidence:.3f})")


@dataclass
class NodeBMetrics:
    roc_auc: float
    logloss: float
    brier: float
    accuracy: float
    recall_tired: float
    precision_tired: float
    f1_tired: float
    recall_active: float
    f1_active: float
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
    thresholds = np.linspace(0.01, 0.99, 200)

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
#  NODE B MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class NodeBModel:
    """Node B: Tired vs Active binary classifier."""

    def __init__(self, sr=22050, duration=5.0, random_state=42, verbose=True):
        self.sr = sr
        self.duration = duration
        self.rs = random_state
        self.verbose = verbose

        self.pipeline: Optional[Pipeline] = None
        self.feature_names: Optional[List[str]] = None
        self.tau_low: float = 0.30
        self.tau_high: float = 0.65
        self.is_fitted: bool = False

    def fit(self, X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names=None) -> NodeBMetrics:
        """Train Node B. y: 1=tired, 0=active."""
        self.feature_names = feature_names

        if self.verbose:
            n_t = np.sum(y_train == 1)
            n_a = np.sum(y_train == 0)
            print(f"  Train: {len(y_train)} ({n_t} tired, {n_a} active)")
            print(f"  Val:   {len(y_val)}")
            print(f"  Test:  {len(y_test)}")

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
            print("  Training Node B ensemble (RF + GB + LR) ...")

        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True

        p_val = self.pipeline.predict_proba(X_val)[:, 1]
        self.tau_low, self.tau_high = _find_thresholds(y_val, p_val)

        if self.verbose:
            print(f"  Thresholds: τ_low={self.tau_low:.3f}, τ_high={self.tau_high:.3f}")

        p_test = self.pipeline.predict_proba(X_test)[:, 1]
        metrics = self._compute_metrics(y_test, p_test, len(y_train), len(y_val))

        if self.verbose:
            print(f"\n  ── Test Results ──")
            print(f"  AUC={metrics.roc_auc:.4f}  Brier={metrics.brier:.4f}  "
                  f"LogLoss={metrics.logloss:.4f}")
            print(f"  Accuracy={metrics.accuracy:.4f}")
            print(f"  Tired:  recall={metrics.recall_tired:.4f}  "
                  f"precision={metrics.precision_tired:.4f}  F1={metrics.f1_tired:.4f}")
            print(f"  Active: recall={metrics.recall_active:.4f}  "
                  f"F1={metrics.f1_active:.4f}")
            print(f"  Uncertain zone: {metrics.uncertain_fraction:.1%}")

        return metrics

    def _compute_metrics(self, y_true, p_pos, n_train=0, n_val=0) -> NodeBMetrics:
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

        return NodeBMetrics(
            roc_auc=roc, logloss=ll, brier=br, accuracy=acc,
            recall_tired=float(rec[1]), precision_tired=float(prec[1]),
            f1_tired=float(f1[1]),
            recall_active=float(rec[0]), f1_active=float(f1[0]),
            tau_low=self.tau_low, tau_high=self.tau_high,
            uncertain_fraction=uncertain,
            n_train=n_train, n_val=n_val, n_test=len(y_true),
        )

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, audio_path: str) -> NodeBResult:
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")

        y = load_audio(audio_path, sr=self.sr, duration=self.duration)
        if y is None:
            return NodeBResult(p_tired=0.5, zone=NodeBZone.UNCERTAIN, confidence=0.5)

        feats = extract_cause_features(y, self.sr)
        if feats is None:
            return NodeBResult(p_tired=0.5, zone=NodeBZone.UNCERTAIN, confidence=0.5)

        names = self.feature_names or sorted(feats.keys())
        x = np.array(
            [float(feats.get(f, 0.0)) for f in names], dtype=float
        ).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        p_tired = float(self.pipeline.predict_proba(x)[0, 1])
        confidence = max(p_tired, 1.0 - p_tired)

        if p_tired >= self.tau_high:
            zone = NodeBZone.TIRED
        elif p_tired >= self.tau_low:
            zone = NodeBZone.UNCERTAIN
        else:
            zone = NodeBZone.ACTIVE

        return NodeBResult(p_tired=p_tired, zone=zone, confidence=confidence)

    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        if self.verbose:
            print(f"  Saved: {filepath}")

    @staticmethod
    def load(filepath: str) -> "NodeBModel":
        with open(filepath, "rb") as f:
            return pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    print("=" * 60)
    print("  Testing nodeB_model.py with synthetic data")
    print("=" * 60)

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

    model = NodeBModel(verbose=True)
    metrics = model.fit(X_train, y_train, X_val, y_val, X_test, y_test,
                        feature_names=[f"feat_{i}" for i in range(97)])

    print(f"\n  ── Verification ──")
    assert metrics.roc_auc > 0.60
    print(f"  ✅ AUC = {metrics.roc_auc:.4f}")

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmp = f.name
    model.save(tmp)
    model2 = NodeBModel.load(tmp)
    p1 = model.predict_proba_batch(X_test)
    p2 = model2.predict_proba_batch(X_test)
    assert np.allclose(p1, p2)
    print(f"  ✅ Save/Load OK")
    os.unlink(tmp)

    r = NodeBResult(p_tired=0.80, zone=NodeBZone.TIRED, confidence=0.80)
    assert abs(r.proba_dict["tired"] - 0.80) < 1e-9
    assert abs(r.proba_dict["active"] - 0.20) < 1e-9
    print(f"  ✅ NodeBResult: {r}")

    print(f"\n{'=' * 60}")
    print(f"  nodeB_model.py — ALL TESTS PASSED ✅")
    print(f"{'=' * 60}")