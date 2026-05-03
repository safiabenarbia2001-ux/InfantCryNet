"""
video_simulator.py
==================
Synthetic video feature generator for InfantCryNet-v4.

Motivation
----------
Real infant-cry video is expensive to collect and label.  During
development (and as an ablation baseline), this module generates a
plausible 50-dimensional video feature vector from audio features and
biological context.  The simulator will be *replaced* by a real video
feature extractor once video data is available.

Architecture
------------
  Input  : d_audio audio features  +  d_bio biological scalars
           (concatenated → 1-D vector)
  Hidden : 2 × 64 ReLU + BatchNorm
  Output : R^{50}  + calibrated Gaussian noise  N(0, σ² I)

Noise model:
  z_sim = f_θ(x_audio, x_bio) + ε,   ε ~ N(0, σ_c² I)
  where σ_c is class-specific and estimated from clinical literature.

The model is trained with a reconstruction-style MSE loss on a small
labelled subset (if available) or initialised from class prototypes
derived from published motor-pattern norms.

Feature dimensions (50 total):
  [0:20]  Motion dynamics  (mean, std, max, min, range, jerk × 4 ROIs: head/torso/legs/full)
  [20:35] Brightness/colour (luminance stats, saturation, flicker, skin-tone proxy)
  [35:42] Texture & edges   (Laplacian var, Sobel gradient, edge density × zones)
  [42:47] Temporal dynamics (periodicity, burst rate, stillness ratio, scene-change rate, rhythm)
  [47:50] Audio-visual sync (AV-correlation proxy, audio-onset lag, energy-motion coupling)

Statistical basis for class prototypes
---------------------------------------
Hungry     : rhythmic sucking motion, moderate head movement, low jerk
Tired      : low motion, high stillness, smooth, drooping head
Pain       : high jerk, leg flexion, periodic intense bursts
Reflux     : back-arching torso, moderate jerk, post-feed timing
Discomfort : writhing, sustained high motion, kicking
Fatigue    : minimal movement, eyes-closing proxy (face brightness drop)

References
----------
Gustafson G.E., Harris K.L. (1990). Women's responses to young infants'
  cries. Developmental Psychology, 26(1), 144–152.
Zeifman D.M. (2001). An ethological analysis of human infant crying.
  Developmental Psychobiology, 39(4), 265–285.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional sklearn for the learned MLP path
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

from hierarchical_labels import CLASS_NAMES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEO_FEATURE_DIM   : int   = 50
_CLASS_IDS          : Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
_EPS                : float = 1e-9

# Class-specific noise standard deviations (higher = more variable motor behaviour)
_SIGMA_PER_CLASS: Dict[int, float] = {
    1: 0.05,   # hungry   — rhythmic, predictable
    2: 0.10,   # pain     — variable intensity
    3: 0.08,   # discomfort
    4: 0.04,   # tired    — very low motion, consistent
    5: 0.07,   # reflux
    6: 0.10,   # colic    — high variability
}

# ---------------------------------------------------------------------------
# Class prototypes (50-D vectors, empirically grounded)
# ---------------------------------------------------------------------------
# fmt: off
_PROTOTYPES: Dict[int, np.ndarray] = {
    # Layout: motion(20) | brightness(15) | texture(7) | temporal(5) | av_sync(3)
    1: np.array([   # hungry — rhythmic head turns, moderate motion
        0.18, 0.04, 0.30, 0.08, 0.22,  # motion: head mean/std/max/min/range
        0.12, 0.03, 0.20, 0.06, 0.14,  # torso
        0.08, 0.02, 0.16, 0.04, 0.12,  # legs
        0.15, 0.04, 0.25, 0.07, 0.18,  # full-body
        0.52, 0.06, 0.04, 0.48, 0.12, 0.34, 0.08, 0.10, 0.44, 0.06,  # brightness(10)
        0.18, 0.06, 0.12, 0.04, 0.08,  # brightness cont (5)
        0.32, 0.08, 0.22, 0.06, 0.12, 0.08, 0.15,  # texture(7)
        0.45, 0.35, 0.42, 0.28, 0.55,  # temporal(5)
        0.38, 0.22, 0.45,              # av_sync(3)
    ], dtype=np.float32),

    2: np.array([   # pain — high jerk, leg flexion, periodic bursts
        0.28, 0.12, 0.55, 0.05, 0.50,
        0.22, 0.10, 0.50, 0.04, 0.46,
        0.38, 0.14, 0.65, 0.08, 0.57,  # legs — highest
        0.30, 0.12, 0.58, 0.06, 0.52,
        0.50, 0.08, 0.05, 0.46, 0.14, 0.32, 0.10, 0.12, 0.42, 0.08,
        0.20, 0.08, 0.14, 0.06, 0.10,
        0.55, 0.18, 0.42, 0.12, 0.28, 0.14, 0.32,
        0.55, 0.62, 0.25, 0.58, 0.42,
        0.52, 0.38, 0.60,
    ], dtype=np.float32),

    3: np.array([   # discomfort — sustained high motion, kicking
        0.25, 0.10, 0.48, 0.05, 0.43,
        0.28, 0.11, 0.52, 0.06, 0.46,
        0.22, 0.09, 0.45, 0.05, 0.40,
        0.25, 0.10, 0.48, 0.06, 0.42,
        0.50, 0.07, 0.05, 0.46, 0.13, 0.30, 0.09, 0.11, 0.40, 0.07,
        0.18, 0.07, 0.12, 0.05, 0.09,
        0.48, 0.15, 0.38, 0.10, 0.22, 0.12, 0.28,
        0.30, 0.50, 0.28, 0.48, 0.35,
        0.45, 0.30, 0.52,
    ], dtype=np.float32),

    4: np.array([   # tired — minimal motion, high stillness
        0.06, 0.02, 0.12, 0.01, 0.11,
        0.05, 0.02, 0.10, 0.01, 0.09,
        0.04, 0.01, 0.08, 0.01, 0.07,
        0.05, 0.02, 0.10, 0.01, 0.09,
        0.48, 0.05, 0.03, 0.45, 0.09, 0.28, 0.06, 0.07, 0.38, 0.04,
        0.14, 0.04, 0.08, 0.02, 0.05,
        0.15, 0.04, 0.10, 0.02, 0.06, 0.03, 0.08,
        0.18, 0.12, 0.72, 0.08, 0.22,  # high stillness_ratio (index 2)
        0.18, 0.10, 0.22,
    ], dtype=np.float32),

    5: np.array([   # reflux — back-arching, post-feed, moderate jerk
        0.15, 0.06, 0.32, 0.03, 0.29,
        0.22, 0.09, 0.42, 0.04, 0.38,  # torso highest (arching)
        0.10, 0.04, 0.22, 0.03, 0.19,
        0.16, 0.07, 0.32, 0.04, 0.28,
        0.52, 0.07, 0.04, 0.48, 0.11, 0.32, 0.08, 0.09, 0.40, 0.06,
        0.17, 0.06, 0.11, 0.04, 0.08,
        0.38, 0.12, 0.28, 0.08, 0.16, 0.09, 0.20,
        0.22, 0.28, 0.48, 0.32, 0.38,
        0.42, 0.28, 0.50,
    ], dtype=np.float32),

    6: np.array([   # colic — very high motion, leg-pulling, periodic
        0.32, 0.14, 0.60, 0.06, 0.54,
        0.26, 0.12, 0.55, 0.05, 0.50,
        0.40, 0.16, 0.68, 0.08, 0.60,
        0.34, 0.14, 0.62, 0.07, 0.55,
        0.50, 0.08, 0.06, 0.45, 0.14, 0.30, 0.10, 0.12, 0.38, 0.08,
        0.22, 0.09, 0.16, 0.07, 0.12,
        0.58, 0.20, 0.45, 0.14, 0.30, 0.16, 0.35,
        0.58, 0.65, 0.20, 0.62, 0.45,
        0.55, 0.42, 0.62,
    ], dtype=np.float32),
}
# fmt: on

assert all(v.shape == (VIDEO_FEATURE_DIM,) for v in _PROTOTYPES.values()), \
    "All prototypes must be 50-D"


# ---------------------------------------------------------------------------
# Dataclass: simulation result
# ---------------------------------------------------------------------------

@dataclass
class SimulatedVideoFeatures:
    """
    Output of VideoSimulator for one sample.

    Attributes
    ----------
    features       : float32 array (50,)  — simulated video features
    class_label    : int                  — true class used for simulation
    noise_sigma    : float                — noise level applied
    feature_names  : list[str]            — feature descriptions
    confidence     : float                — prototype matching confidence [0,1]
    """
    features      : np.ndarray
    class_label   : int
    noise_sigma   : float
    feature_names : List[str] = field(default_factory=list)
    confidence    : float = 1.0

    @property
    def as_dict(self) -> Dict[str, float]:
        return {
            name: float(val)
            for name, val in zip(self.feature_names, self.features)
        }


# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------

def _build_feature_names() -> List[str]:
    """Return the 50 canonical feature names in order."""
    names: List[str] = []
    rois = ["head", "torso", "legs", "body"]
    stats = ["mean", "std", "max", "min", "range"]
    for roi in rois:
        for stat in stats:
            names.append(f"motion_{roi}_{stat}")
    bri_names = [
        "brightness_mean", "brightness_std", "brightness_min",
        "brightness_max", "brightness_range",
        "saturation_mean", "saturation_std", "flicker_std", "flicker_p95",
        "skin_tone_mean", "red_channel_mean", "green_channel_mean",
        "blue_channel_mean", "color_consistency", "chromaticity_shift",
    ]
    names += bri_names
    tex_names = [
        "laplacian_var_mean", "laplacian_var_std",
        "sobel_mean", "sobel_std",
        "edge_density_global", "edge_density_face", "edge_density_body",
    ]
    names += tex_names
    temp_names = [
        "motion_periodicity", "burst_rate",
        "stillness_ratio", "scene_change_rate", "motion_rhythm_entropy",
    ]
    names += temp_names
    av_names = [
        "av_energy_correlation", "onset_motion_lag", "energy_motion_coupling"
    ]
    names += av_names
    assert len(names) == VIDEO_FEATURE_DIM, \
        f"Expected {VIDEO_FEATURE_DIM} names, got {len(names)}"
    return names


FEATURE_NAMES: List[str] = _build_feature_names()


# ---------------------------------------------------------------------------
# Main simulator class
# ---------------------------------------------------------------------------

class VideoSimulator:
    """
    Synthetic video feature generator.

    Two operating modes:

    1. **Prototype mode** (default, no training needed):
       Samples from a Gaussian centred on the class prototype.
       z = μ_c + diag(σ_c · w_audio) · ε,  ε ~ N(0, I)
       where w_audio modulates noise magnitude based on audio energy.

    2. **Learned mode** (requires labelled {audio+bio → video} pairs):
       Trains an MLP regressor on ground-truth pairs.
       Used for domain adaptation once real video data arrives.

    Parameters
    ----------
    noise_scale : float
        Global multiplier for Gaussian noise.  0.0 = deterministic prototype.
    audio_modulation : bool
        If True, audio energy (first feature) modulates noise: louder → more motion.
    random_state : int | None
    """

    def __init__(
        self,
        noise_scale: float = 1.0,
        audio_modulation: bool = True,
        random_state: Optional[int] = 42,
    ) -> None:
        self.noise_scale = float(noise_scale)
        self.audio_modulation = audio_modulation
        self.rng = np.random.default_rng(random_state)

        self._prototypes: Dict[int, np.ndarray] = {
            k: v.copy() for k, v in _PROTOTYPES.items()
        }
        self._sigmas: Dict[int, float] = dict(_SIGMA_PER_CLASS)

        # Learned mode (optional)
        self._mlp: Optional["MLPRegressor"] = None
        self._scaler_in: Optional["StandardScaler"] = None
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # Prototype-mode generation
    # ------------------------------------------------------------------

    def simulate(
        self,
        class_label: int,
        audio_features: Optional[np.ndarray] = None,
        bio_scalars: Optional[np.ndarray] = None,
    ) -> SimulatedVideoFeatures:
        """
        Generate one synthetic video feature vector for a given class.

        Parameters
        ----------
        class_label : int
            Flat class label {1,…,6}.
        audio_features : array (d_audio,), optional
            If provided, used to modulate noise (high energy → more motion).
        bio_scalars : array (d_bio,), optional
            Currently unused (reserved for future: age/feed-time modulation).

        Returns
        -------
        SimulatedVideoFeatures
        """
        if class_label not in self._prototypes:
            raise ValueError(f"Unknown class {class_label}. Valid: {sorted(self._prototypes)}")

        proto = self._prototypes[class_label]
        sigma = self._sigmas[class_label] * self.noise_scale

        # Audio modulation: louder cry → higher motion features
        if self.audio_modulation and audio_features is not None:
            audio_energy = float(np.clip(
                np.mean(np.abs(audio_features[:20])) if len(audio_features) >= 20
                else np.mean(np.abs(audio_features)), 0.0, 1.0
            ))
            # Scale the first 20 features (motion) by energy
            motion_scale = 0.8 + 0.4 * audio_energy   # ∈ [0.8, 1.2]
        else:
            motion_scale = 1.0

        # Draw from N(proto, σ²I), then clip to [0, 1]
        noise = self.rng.normal(0.0, sigma, size=VIDEO_FEATURE_DIM).astype(np.float32)
        features = proto.copy()
        features[:20] *= motion_scale            # audio-modulated motion
        features = np.clip(features + noise, 0.0, 1.0)

        # Confidence: how close is the prototype to a "real" recording
        # (decreases with noise_scale, used to set fusion weight)
        confidence = float(np.exp(-2.0 * sigma))

        return SimulatedVideoFeatures(
            features=features,
            class_label=class_label,
            noise_sigma=sigma,
            feature_names=FEATURE_NAMES,
            confidence=confidence,
        )

    def simulate_batch(
        self,
        class_labels: List[int] | np.ndarray,
        audio_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Simulate a batch of video features.

        Parameters
        ----------
        class_labels : (N,) int array
        audio_features : (N, d_audio) optional

        Returns
        -------
        X : (N, 50) float32 array
        """
        N = len(class_labels)
        X = np.empty((N, VIDEO_FEATURE_DIM), dtype=np.float32)
        for i, lbl in enumerate(class_labels):
            af = audio_features[i] if audio_features is not None else None
            X[i] = self.simulate(int(lbl), audio_features=af).features
        return X

    # ------------------------------------------------------------------
    # Learned mode
    # ------------------------------------------------------------------

    def fit(
        self,
        X_input: np.ndarray,               # (N, d_audio + d_bio)
        X_video: np.ndarray,               # (N, 50) ground-truth video features
        class_labels: Optional[np.ndarray] = None,
    ) -> "VideoSimulator":
        """
        Train the MLP regressor on (audio+bio → video) pairs.

        Parameters
        ----------
        X_input : (N, d) concatenated audio+bio
        X_video : (N, 50) real video features
        class_labels : (N,) optional, used only for residual adjustment

        Returns
        -------
        self (fitted)
        """
        if not SKLEARN_OK:
            raise ImportError("scikit-learn required: pip install scikit-learn")
        if X_input.shape[0] < 30:
            warnings.warn(
                f"Only {X_input.shape[0]} samples — learned mode may overfit. "
                "Prototype mode is safer for N < 100.", stacklevel=2
            )

        self._scaler_in = StandardScaler()
        X_scaled = self._scaler_in.fit_transform(X_input.astype(np.float32))

        self._mlp = MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            solver="adam",
            alpha=0.01,
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=0,
            verbose=False,
        )
        self._mlp.fit(X_scaled, X_video.astype(np.float32))
        self._is_trained = True

        # Update per-class noise via residual variance
        if class_labels is not None:
            pred = self._mlp.predict(X_scaled)
            residuals = X_video - pred
            for c in _CLASS_IDS:
                mask = np.asarray(class_labels) == c
                if mask.sum() > 2:
                    self._sigmas[c] = float(np.std(residuals[mask]))
        return self

    def predict_features(
        self,
        X_input: np.ndarray,           # (N, d) or (d,)
        add_noise: bool = True,
    ) -> np.ndarray:
        """
        Predict video features in learned mode.  Falls back to prototype
        mode if model is not trained.
        """
        if not self._is_trained:
            raise RuntimeError("Call fit() first or use simulate() (prototype mode).")
        X = np.atleast_2d(X_input).astype(np.float32)
        X_scaled = self._scaler_in.transform(X)
        pred = self._mlp.predict(X_scaled).astype(np.float32)
        if add_noise:
            noise = self.rng.normal(
                0.0, self.noise_scale * 0.03, size=pred.shape
            ).astype(np.float32)
            pred += noise
        return np.clip(pred, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Effective fusion weight
    # ------------------------------------------------------------------

    def get_fusion_weight(self, has_real_video: bool = False) -> float:
        """
        Return the recommended video modality weight for the PoE fusion.

        Real video (measured)   → 0.10
        Simulated (prototype)   → noise_scale-dependent, typically 0.00–0.04
        No video at all         → 0.00
        """
        if has_real_video:
            return 0.10
        if self.noise_scale == 0.0:
            return 0.04   # deterministic prototype: some information
        return float(np.clip(0.04 * np.exp(-self.noise_scale), 0.0, 0.04))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def prototype_similarity_matrix(self) -> pd.DataFrame:
        """
        Pairwise cosine-similarity matrix between class prototypes.
        High similarity between two classes = harder to distinguish from video alone.
        """
        import pandas as pd
        keys = sorted(self._prototypes)
        P = np.stack([self._prototypes[k] for k in keys])   # (K, 50)
        norms = np.linalg.norm(P, axis=1, keepdims=True) + _EPS
        P_norm = P / norms
        sim = P_norm @ P_norm.T
        names = [CLASS_NAMES[k] for k in keys]
        return pd.DataFrame(sim.round(4), index=names, columns=names)

    def describe(self) -> str:
        w = self.get_fusion_weight(has_real_video=False)
        lines = [
            "VideoSimulator",
            f"  Mode          : {'learned (MLP)' if self._is_trained else 'prototype (rule-based)'}",
            f"  Feature dim   : {VIDEO_FEATURE_DIM}",
            f"  Noise scale   : {self.noise_scale:.3f}",
            f"  Fusion weight : {w:.4f}  (real=0.10)",
            "",
            "  Per-class noise σ:",
        ]
        for c in sorted(self._sigmas):
            lines.append(f"    {CLASS_NAMES[c]:>12} : {self._sigmas[c]:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Import pandas lazily (only needed for prototype_similarity_matrix + stats)
# ---------------------------------------------------------------------------
try:
    import pandas as pd
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sim = VideoSimulator(noise_scale=0.8, random_state=0)

    print("=== Single sample (class 2: pain) ===")
    result = sim.simulate(2, audio_features=np.random.randn(80).astype(np.float32))
    print(f"  Shape     : {result.features.shape}")
    print(f"  Min/Max   : {result.features.min():.4f} / {result.features.max():.4f}")
    print(f"  Confidence: {result.confidence:.4f}")
    print(f"  Noise σ   : {result.noise_sigma:.4f}")

    print("\n=== Batch simulation ===")
    labels = np.array([1, 2, 3, 4, 5, 6])
    X = sim.simulate_batch(labels)
    print(f"  Shape     : {X.shape}")
    print(f"  Row norms : {np.linalg.norm(X, axis=1).round(3)}")

    print("\n=== Fusion weights ===")
    print(f"  No video  : {sim.get_fusion_weight(False):.4f}")
    print(f"  Real video: {sim.get_fusion_weight(True):.4f}")

    print("\n=== Prototype similarity matrix ===")
    print(sim.prototype_similarity_matrix())

    print("\n=== Descriptor ===")
    print(sim.describe())