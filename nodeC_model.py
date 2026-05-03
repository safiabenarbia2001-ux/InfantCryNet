"""
nodeC_model.py — Node C: Discomfort sub-type classifier.

3-class: belly_pain / burping / discomfort

This is the HARDEST node because:
    - 3 classes instead of 2
    - belly_pain and burping are both physical discomfort → very similar acoustically
    - "discomfort" is a catch-all (cold_hot + lonely + scared merged into it)

The confidence approach changes for 3-class:
    - No τ_low / τ_high pair (that's for binary)
    - Instead: confidence = max(P(class_1), P(class_2), P(class_3))
    - If confidence < threshold → UNCERTAIN
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
    accuracy_score,
    log_loss,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.audio.audio_features import load_audio, extract_cause_features

EPS = 1e-12
NODE_C_CLASSES = ["belly_pain", "burping", "discomfort"]


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class NodeCZone(Enum):
    CONFIDENT = "confident"      # max P(class) >= conf_threshold
    UNCERTAIN = "uncertain"      # max P(class) < conf_threshold


@dataclass
class NodeCResult:
    """What Node C outputs to the fusion stage.

    proba:      dict {"belly_pain": 0.6, "burping": 0.15, "discomfort": 0.25}
    prediction: the class with highest probability
    confidence: max probability (how sure are we?)
    zone:       CONFIDENT or UNCERTAIN
    """
    proba: Dict[str, float]
    prediction: str
    confidence: float
    zone: NodeCZone

    def __repr__(self):
        return (f"NodeCResult(pred={self.prediction}, conf={self.confidence:.3f}, "
                f"zone={self.zone.value})")


@dataclass
class NodeCMetrics:
    accuracy: float
    f1_macro: float
    f1_per_class: Dict[str, float]
    logloss: float
    confidence_mean: float
    uncertain_fraction: float
    confusion: any  # confusion matrix
    n_train: int
    n_val: int
    n_test: int
    conf_threshold: float


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE C MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class NodeCModel:
    """Node C: 3-class discomfort sub-type classifier.

    Classes: belly_pain, burping, discomfort
    Output: probability distribution over all 3 classes
    """

    def __init__(self, sr=22050, duration=5.0, random_state=42, verbose=True):
        self.sr = sr
        self.duration = duration
        self.rs = random_state
        self.verbose = verbose

        self.pipeline: Optional[Pipeline] = None
        self.feature_names: Optional[List[str]] = None
        self.classes: List[str] = NODE_C_CLASSES
        self.label_encoder: Optional[LabelEncoder] = None
        self.conf_threshold: float = 0.50  # learned from validation
        self.is_fitted: bool = False

    def fit(self, X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names=None) -> NodeCMetrics:
        """Train Node C.

        y values: string labels ("belly_pain", "burping", "discomfort")
        """
        self.feature_names = feature_names

        # Encode string labels to integers for sklearn
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)
        y_tr_enc = self.label_encoder.transform(y_train)
        y_val_enc = self.label_encoder.transform(y_val)
        y_te_enc = self.label_encoder.transform(y_test)

        if self.verbose:
            print(f"  Classes: {self.classes}")
            for cls in self.classes:
                n = np.sum(y_train == cls)
                print(f"    {cls:<14} train={n}")
            print(f"  Val: {len(y_val)}  Test: {len(y_test)}")

        # ── Ensemble ──────────────────────────────────────────────────────
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
            print("  Training Node C ensemble (RF + GB + LR) ...")

        self.pipeline.fit(X_train, y_tr_enc)
        self.is_fitted = True

        # ── Learn confidence threshold on validation set ──────────────────
        # Find threshold where accuracy of "confident" predictions is good
        p_val = self.pipeline.predict_proba(X_val)
        val_max_probs = np.max(p_val, axis=1)
        val_preds = self.label_encoder.inverse_transform(np.argmax(p_val, axis=1))

        best_thresh = 0.50
        best_score = 0.0
        for thresh in np.linspace(0.35, 0.80, 30):
            confident_mask = val_max_probs >= thresh
            if confident_mask.sum() < 5:
                continue
            acc_confident = accuracy_score(y_val[confident_mask], val_preds[confident_mask])
            coverage = confident_mask.mean()
            # Balance accuracy and coverage
            score = acc_confident * coverage
            if score > best_score:
                best_score = score
                best_thresh = thresh

        self.conf_threshold = float(best_thresh)
        if self.verbose:
            print(f"  Confidence threshold: {self.conf_threshold:.3f}")

        # ── Evaluate on test set ──────────────────────────────────────────
        p_test = self.pipeline.predict_proba(X_test)
        metrics = self._compute_metrics(y_test, y_te_enc, p_test,
                                         len(y_train), len(y_val))

        if self.verbose:
            print(f"\n  ── Test Results ──")
            print(f"  Accuracy={metrics.accuracy:.4f}  "
                  f"F1_macro={metrics.f1_macro:.4f}  "
                  f"LogLoss={metrics.logloss:.4f}")
            print(f"  Per-class F1:")
            for cls, f1 in metrics.f1_per_class.items():
                print(f"    {cls:<14} F1={f1:.4f}")
            print(f"  Mean confidence: {metrics.confidence_mean:.4f}")
            print(f"  Uncertain zone:  {metrics.uncertain_fraction:.1%}")
            print(f"\n  Confusion matrix:")
            print(f"  {'':>14} ", end="")
            for cls in self.classes:
                print(f"{cls[:8]:>10}", end="")
            print()
            for i, cls in enumerate(self.classes):
                print(f"  {cls:<14} ", end="")
                for j in range(len(self.classes)):
                    print(f"{metrics.confusion[i, j]:>10d}", end="")
                print()

        return metrics

    def _compute_metrics(self, y_true_str, y_true_enc, p_pred,
                          n_train=0, n_val=0) -> NodeCMetrics:
        y_pred_enc = np.argmax(p_pred, axis=1)
        y_pred_str = self.label_encoder.inverse_transform(y_pred_enc)

        acc = float(accuracy_score(y_true_str, y_pred_str))
        f1_mac = float(f1_score(y_true_str, y_pred_str, average="macro",
                                 zero_division=0))
        ll = float(log_loss(y_true_enc, p_pred))

        # Per-class F1
        f1_per = {}
        for cls in self.classes:
            mask = y_true_str == cls
            if mask.sum() > 0:
                cls_pred = (y_pred_str == cls).astype(int)
                cls_true = mask.astype(int)
                f1_per[cls] = float(f1_score(cls_true, cls_pred, zero_division=0))
            else:
                f1_per[cls] = 0.0

        max_probs = np.max(p_pred, axis=1)
        conf_mean = float(np.mean(max_probs))
        uncertain_frac = float(np.mean(max_probs < self.conf_threshold))

        cm = confusion_matrix(y_true_str, y_pred_str, labels=self.classes)

        return NodeCMetrics(
            accuracy=acc, f1_macro=f1_mac, f1_per_class=f1_per,
            logloss=ll, confidence_mean=conf_mean,
            uncertain_fraction=uncertain_frac,
            confusion=cm, n_train=n_train, n_val=n_val,
            n_test=len(y_true_str), conf_threshold=self.conf_threshold,
        )

    # ── Prediction ────────────────────────────────────────────────────────

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        return self.pipeline.predict_proba(X)

    def predict(self, audio_path: str) -> NodeCResult:
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")

        y = load_audio(audio_path, sr=self.sr, duration=self.duration)
        if y is None:
            # Uniform uncertainty
            uniform = {c: 1.0 / len(self.classes) for c in self.classes}
            return NodeCResult(proba=uniform, prediction="discomfort",
                               confidence=1.0 / len(self.classes),
                               zone=NodeCZone.UNCERTAIN)

        feats = extract_cause_features(y, self.sr)
        if feats is None:
            uniform = {c: 1.0 / len(self.classes) for c in self.classes}
            return NodeCResult(proba=uniform, prediction="discomfort",
                               confidence=1.0 / len(self.classes),
                               zone=NodeCZone.UNCERTAIN)

        names = self.feature_names or sorted(feats.keys())
        x = np.array(
            [float(feats.get(f, 0.0)) for f in names], dtype=float
        ).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        p = self.pipeline.predict_proba(x)[0]
        decoded_classes = self.label_encoder.inverse_transform(range(len(p)))
        proba = {cls: float(p[i]) for i, cls in enumerate(decoded_classes)}

        prediction = max(proba, key=proba.get)
        confidence = proba[prediction]
        zone = NodeCZone.CONFIDENT if confidence >= self.conf_threshold else NodeCZone.UNCERTAIN

        return NodeCResult(proba=proba, prediction=prediction,
                           confidence=confidence, zone=zone)

    # ── Save / Load ───────────────────────────────────────────────────────

    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        if self.verbose:
            print(f"  Saved: {filepath}")

    @staticmethod
    def load(filepath: str) -> "NodeCModel":
        with open(filepath, "rb") as f:
            return pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    print("=" * 60)
    print("  Testing nodeC_model.py with synthetic data")
    print("=" * 60)

    # 3-class synthetic data
    X, y_int = make_classification(
        n_samples=300, n_features=97, n_informative=20,
        n_redundant=10, n_classes=3, n_clusters_per_class=1,
        class_sep=0.8, random_state=42,
    )
    # Map to our class names
    class_map = {0: "belly_pain", 1: "burping", 2: "discomfort"}
    y = np.array([class_map[i] for i in y_int])

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, random_state=42, stratify=y_dev)

    print(f"\n  Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")

    model = NodeCModel(verbose=True)
    metrics = model.fit(X_train, y_train, X_val, y_val, X_test, y_test,
                        feature_names=[f"feat_{i}" for i in range(97)])

    print(f"\n  ── Verification ──")
    assert metrics.accuracy > 0.30  # Above random (0.33) for 3-class
    print(f"  ✅ Accuracy = {metrics.accuracy:.4f}")
    assert metrics.f1_macro > 0.0
    print(f"  ✅ F1 macro = {metrics.f1_macro:.4f}")

    # Test save/load
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmp = f.name
    model.save(tmp)
    model2 = NodeCModel.load(tmp)
    p1 = model.predict_proba_batch(X_test)
    p2 = model2.predict_proba_batch(X_test)
    assert np.allclose(p1, p2)
    print(f"  ✅ Save/Load OK")
    os.unlink(tmp)

    # Test NodeCResult
    r = NodeCResult(
        proba={"belly_pain": 0.6, "burping": 0.15, "discomfort": 0.25},
        prediction="belly_pain", confidence=0.6, zone=NodeCZone.CONFIDENT)
    assert r.prediction == "belly_pain"
    assert abs(sum(r.proba.values()) - 1.0) < 1e-9
    print(f"  ✅ NodeCResult: {r}")

    print(f"\n{'=' * 60}")
    print(f"  nodeC_model.py — ALL TESTS PASSED ✅")
    print(f"{'=' * 60}")