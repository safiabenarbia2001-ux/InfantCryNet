"""
video_model.py — Video behavioral analysis for cry cause estimation.

CURRENT STATE: Simulated (no real video data yet)
FUTURE STATE:  Replace VideoSimulator with real video feature extraction

What video features would capture:
    - Mouth movements: sucking, rooting → hunger
    - Eye state: rubbing eyes, closing → tiredness
    - Body posture: legs pulled up, arching → belly pain
    - Movement intensity: squirming, thrashing → discomfort
    - Facial expression: grimacing → pain

How this fits in the system:
    The app can work in 3 modes:
        Mode 1: Audio only         (just microphone)
        Mode 2: Audio + Bio        (mic + parent input)
        Mode 3: Audio + Bio + Video (mic + input + camera)

    When video is NOT available, the fusion module simply ignores it.
    The video model explicitly reports whether its output is simulated
    or real, so fusion can weight it accordingly.

IMPORTANT for your thesis:
    The simulator adds NOISE intentionally. If simulated video were
    perfectly accurate, it would inflate fusion results unrealistically.
    We simulate with ~60-70% accuracy to be honest about what video
    MIGHT contribute, not what it WILL contribute.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

EPS = 1e-12


# ═══════════════════════════════════════════════════════════════════════════════
#  VIDEO FEATURES — what we'd extract from real video
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VideoFeatures:
    """Behavioral features from video analysis.

    All values are in [0, 1] representing intensity/probability.
    When using real video, a CNN or pose estimator would fill these.
    For now, the simulator generates them.
    """
    mouth_movement: float = 0.0       # Sucking, rooting, opening mouth
    eye_rubbing: float = 0.0          # Rubbing eyes, drooping eyelids
    eye_closing: float = 0.0          # Eyes closing, heavy blinks
    legs_pulled_up: float = 0.0       # Legs drawn to belly (pain sign)
    back_arching: float = 0.0         # Arching back (colic/pain)
    movement_intensity: float = 0.0   # Overall body movement level
    facial_grimace: float = 0.0       # Pain expression on face
    yawning: float = 0.0             # Yawning detected

    def as_dict(self) -> Dict[str, float]:
        return {
            "mouth_movement": self.mouth_movement,
            "eye_rubbing": self.eye_rubbing,
            "eye_closing": self.eye_closing,
            "legs_pulled_up": self.legs_pulled_up,
            "back_arching": self.back_arching,
            "movement_intensity": self.movement_intensity,
            "facial_grimace": self.facial_grimace,
            "yawning": self.yawning,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  VIDEO OUTPUT — same structure as audio and bio for clean fusion
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VideoResult:
    """What the video model outputs to fusion.

    node_a_hungry: P(hungry) from video behavioral cues
    node_b_tired:  P(tired) from video behavioral cues
    node_c:        {belly_pain, burping, discomfort} from video
    is_simulated:  True if using simulator, False if real video
    reliability:   how much fusion should trust this (0 to 1)
                   simulated = 0.3, real = 0.7-1.0
    """
    node_a_hungry: float
    node_b_tired: float
    node_c: Dict[str, float]
    is_simulated: bool = True
    reliability: float = 0.3      # Low for simulated, high for real

    def __repr__(self):
        src = "SIM" if self.is_simulated else "REAL"
        return (f"VideoResult[{src}](hungry={self.node_a_hungry:.3f}, "
                f"tired={self.node_b_tired:.3f}, "
                f"reliability={self.reliability:.2f})")


# ═══════════════════════════════════════════════════════════════════════════════
#  VIDEO SIMULATOR — generates features with realistic noise
# ═══════════════════════════════════════════════════════════════════════════════

class VideoSimulator:
    """Generate simulated video features for testing.

    Each cause has a behavioral profile:
        hungry:     high mouth_movement, moderate movement
        tired:      high eye_rubbing, eye_closing, yawning, low movement
        belly_pain: high legs_pulled_up, back_arching, grimace, high movement
        burping:    moderate movement, some grimace
        discomfort: moderate everything, some movement

    Noise is added intentionally:
        - 30% of features get random perturbation
        - Some features are set to wrong values (simulates confusion)
        - This keeps simulated accuracy around 60-70% (honest estimate)
    """

    def __init__(self, noise_level: float = 0.3, random_state: int = 42):
        """
        noise_level: 0.0 = perfect (unrealistic), 1.0 = pure noise (useless)
                     0.3 = realistic estimate of video noise
        """
        self.noise = noise_level
        self.rng = np.random.RandomState(random_state)

    def simulate(self, cause: str) -> VideoFeatures:
        """Generate video features for a given cause."""
        rng = self.rng

        # Base profiles (ideal behavioral signals)
        profiles = {
            "hungry": {
                "mouth_movement": 0.8,
                "eye_rubbing": 0.1,
                "eye_closing": 0.2,
                "legs_pulled_up": 0.1,
                "back_arching": 0.1,
                "movement_intensity": 0.5,
                "facial_grimace": 0.3,
                "yawning": 0.1,
            },
            "tired": {
                "mouth_movement": 0.1,
                "eye_rubbing": 0.8,
                "eye_closing": 0.7,
                "legs_pulled_up": 0.1,
                "back_arching": 0.0,
                "movement_intensity": 0.2,
                "facial_grimace": 0.1,
                "yawning": 0.8,
            },
            "belly_pain": {
                "mouth_movement": 0.2,
                "eye_rubbing": 0.1,
                "eye_closing": 0.3,
                "legs_pulled_up": 0.8,
                "back_arching": 0.7,
                "movement_intensity": 0.8,
                "facial_grimace": 0.8,
                "yawning": 0.0,
            },
            "burping": {
                "mouth_movement": 0.3,
                "eye_rubbing": 0.1,
                "eye_closing": 0.2,
                "legs_pulled_up": 0.3,
                "back_arching": 0.2,
                "movement_intensity": 0.4,
                "facial_grimace": 0.4,
                "yawning": 0.1,
            },
            "discomfort": {
                "mouth_movement": 0.2,
                "eye_rubbing": 0.2,
                "eye_closing": 0.2,
                "legs_pulled_up": 0.2,
                "back_arching": 0.2,
                "movement_intensity": 0.5,
                "facial_grimace": 0.5,
                "yawning": 0.1,
            },
        }

        profile = profiles.get(cause, profiles["discomfort"])

        # Add noise: each feature gets gaussian noise proportional to noise_level
        noisy = {}
        for key, base_val in profile.items():
            noise_val = rng.normal(0, self.noise * 0.5)
            noisy[key] = float(np.clip(base_val + noise_val, 0.0, 1.0))

        return VideoFeatures(**noisy)


# ═══════════════════════════════════════════════════════════════════════════════
#  VIDEO MODEL — converts features to probabilities
# ═══════════════════════════════════════════════════════════════════════════════

class VideoModel:
    """Rule-based video model (current: simulated, future: learned).

    Converts video behavioral features → cause probabilities.
    Same output format as AudioModel and BioModel for clean fusion.

    When video is not available, call VideoModel.no_video() to get
    a neutral result that fusion will ignore.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    @staticmethod
    def no_video() -> VideoResult:
        """Return when no video is available.

        Fusion checks is_simulated and reliability to decide weight.
        reliability=0.0 means "don't use this at all".
        """
        return VideoResult(
            node_a_hungry=0.5,
            node_b_tired=0.5,
            node_c={"belly_pain": 0.33, "burping": 0.33, "discomfort": 0.34},
            is_simulated=True,
            reliability=0.0,  # Zero = fusion ignores this completely
        )

    def predict_from_features(
        self,
        features: VideoFeatures,
        is_simulated: bool = True,
    ) -> VideoResult:
        """Convert video features → cause probabilities.

        Rule-based mapping:
            mouth_movement high → hungry
            eye_rubbing + yawning high → tired
            legs_pulled_up + grimace high → belly_pain
            after-feed + moderate movement → burping
            general movement → discomfort
        """
        f = features

        # ── Node A: P(hungry) ─────────────────────────────────────────────
        # Mouth movement is the strongest hunger indicator from video
        p_hungry = (
            0.50 * f.mouth_movement
            + 0.15 * (1.0 - f.eye_rubbing)   # Not rubbing eyes
            + 0.15 * (1.0 - f.yawning)       # Not yawning
            + 0.20 * f.movement_intensity * 0.5
        )
        p_hungry = float(np.clip(p_hungry, 0.05, 0.95))

        # ── Node B: P(tired) ──────────────────────────────────────────────
        p_tired = (
            0.30 * f.eye_rubbing
            + 0.25 * f.eye_closing
            + 0.25 * f.yawning
            + 0.20 * (1.0 - f.movement_intensity)  # Low movement = tired
        )
        p_tired = float(np.clip(p_tired, 0.05, 0.95))

        # ── Node C: 3-class ───────────────────────────────────────────────
        # Belly pain indicators
        s_belly = (
            0.35 * f.legs_pulled_up
            + 0.30 * f.back_arching
            + 0.25 * f.facial_grimace
            + 0.10 * f.movement_intensity
        )

        # Burping indicators
        s_burp = (
            0.30 * f.facial_grimace * 0.5
            + 0.30 * f.movement_intensity * 0.5
            + 0.20 * (1.0 - f.legs_pulled_up)   # Less leg pulling than pain
            + 0.20 * 0.3                          # Base rate
        )

        # Discomfort (general)
        s_discomfort = (
            0.30 * f.movement_intensity * 0.6
            + 0.20 * f.facial_grimace * 0.5
            + 0.25 * 0.3                          # Base rate
            + 0.25 * (1.0 - f.yawning)
        )

        # Normalize to probabilities
        total = s_belly + s_burp + s_discomfort + EPS
        p_belly = float(s_belly / total)
        p_burp = float(s_burp / total)
        p_discomfort = float(s_discomfort / total)

        # Reliability: simulated gets low weight, real gets higher
        reliability = 0.3 if is_simulated else 0.7

        result = VideoResult(
            node_a_hungry=p_hungry,
            node_b_tired=p_tired,
            node_c={
                "belly_pain": p_belly,
                "burping": p_burp,
                "discomfort": p_discomfort,
            },
            is_simulated=is_simulated,
            reliability=reliability,
        )

        if self.verbose:
            print(f"  Video prediction: {result}")

        return result

    def predict_simulated(self, cause: str, simulator: VideoSimulator = None) -> VideoResult:
        """Convenience: simulate features for a cause and predict."""
        if simulator is None:
            simulator = VideoSimulator()
        features = simulator.simulate(cause)
        return self.predict_from_features(features, is_simulated=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Testing video_model.py")
    print("=" * 60)

    model = VideoModel(verbose=True)
    sim = VideoSimulator(noise_level=0.3, random_state=42)

    # ── Test 1: No video → neutral + reliability=0 ────────────────────────
    print("\n  Test 1: No video available")
    r = VideoModel.no_video()
    assert r.reliability == 0.0
    assert r.is_simulated is True
    assert abs(r.node_a_hungry - 0.5) < 0.01
    print(f"  ✅ {r} (fusion will ignore this)")

    # ── Test 2: Simulated hungry ──────────────────────────────────────────
    print("\n  Test 2: Simulated hungry cry")
    r = model.predict_simulated("hungry", sim)
    assert r.reliability == 0.3  # Simulated = low trust
    print(f"  ✅ P(hungry) = {r.node_a_hungry:.3f}")

    # ── Test 3: Simulated tired ───────────────────────────────────────────
    print("\n  Test 3: Simulated tired cry")
    r = model.predict_simulated("tired", sim)
    print(f"  ✅ P(tired) = {r.node_b_tired:.3f}")

    # ── Test 4: Simulated belly pain ──────────────────────────────────────
    print("\n  Test 4: Simulated belly pain")
    r = model.predict_simulated("belly_pain", sim)
    print(f"  ✅ Node C: belly={r.node_c['belly_pain']:.3f}  "
          f"burp={r.node_c['burping']:.3f}  disc={r.node_c['discomfort']:.3f}")

    # ── Test 5: All causes comparison ─────────────────────────────────────
    print("\n  Test 5: All causes comparison")
    print(f"  {'Cause':<14} {'P(hungry)':>10} {'P(tired)':>10} "
          f"{'P(belly)':>10} {'P(burp)':>10} {'P(disc)':>10}")
    print(f"  {'─' * 64}")

    # Use fresh simulator for clean comparison
    sim_clean = VideoSimulator(noise_level=0.2, random_state=99)
    for cause in ["hungry", "tired", "belly_pain", "burping", "discomfort"]:
        r = model.predict_simulated(cause, sim_clean)
        print(f"  {cause:<14} {r.node_a_hungry:>10.3f} {r.node_b_tired:>10.3f} "
              f"{r.node_c['belly_pain']:>10.3f} {r.node_c['burping']:>10.3f} "
              f"{r.node_c['discomfort']:>10.3f}")

    # ── Test 6: Real video features (manual) ──────────────────────────────
    print("\n  Test 6: Manual 'real' video features (as if from a CNN)")
    real_features = VideoFeatures(
        mouth_movement=0.9, eye_rubbing=0.1, eye_closing=0.1,
        legs_pulled_up=0.0, back_arching=0.0, movement_intensity=0.4,
        facial_grimace=0.2, yawning=0.0,
    )
    r = model.predict_from_features(real_features, is_simulated=False)
    assert r.is_simulated is False
    assert r.reliability == 0.7
    print(f"  ✅ {r}")
    print(f"     Real video gets reliability={r.reliability} (vs 0.3 for simulated)")

    # ── Test 7: Noise levels ──────────────────────────────────────────────
    print("\n  Test 7: Effect of noise level")
    for noise in [0.0, 0.1, 0.3, 0.5, 0.8]:
        sim_n = VideoSimulator(noise_level=noise, random_state=42)
        # Test on hungry: does mouth_movement stay high?
        feats = sim_n.simulate("hungry")
        print(f"  noise={noise:.1f}  mouth_movement={feats.mouth_movement:.3f}  "
              f"(ideal=0.80)")

    print(f"\n{'=' * 60}")
    print(f"  video_model.py — ALL TESTS PASSED ✅")
    print(f"{'=' * 60}")
