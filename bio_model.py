"""
bio_model.py — Biological context model for cry cause estimation.

WHERE THE DATA COMES FROM:
    In the app, the PARENT enters this information:
        - "Last fed 2 hours ago"
        - "Baby is 8 weeks old"
        - "Temperature is 37.5°C"
        - "Last diaper change 1.5 hours ago"
        - "Awake for 90 minutes"

    This model converts those inputs into probabilities:
        P(hungry), P(tired), P(belly_pain), P(burping), P(discomfort)

WHY RULE-BASED (not machine learning):
    1. We have NO real labeled bio data to train on
    2. Medical guidelines for infant feeding/sleep ARE the ground truth
    3. Rules are transparent — a parent can understand WHY
    4. Rules work from day 1 (no training needed)
    5. Easy to update when medical guidelines change

AGE-ADAPTIVE:
    A 2-week-old feeds every 1.5h and sleeps 16h/day.
    A 6-month-old feeds every 3-4h and sleeps 12h/day.
    The model adjusts all thresholds based on age.
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
#  BIO INPUT — what the parent enters in the app
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BioInput:
    """Context information entered by the parent.

    All fields are optional — the model handles missing data gracefully.
    Missing = the parent didn't fill it in → model uses neutral prior.
    """
    minutes_since_feed: Optional[float] = None   # How long since last feeding
    minutes_since_sleep: Optional[float] = None   # How long since baby last slept
    minutes_awake: Optional[float] = None         # How long baby has been awake
    minutes_since_diaper: Optional[float] = None  # How long since diaper change
    temp_c: Optional[float] = None                # Baby's temperature in Celsius
    age_weeks: Optional[float] = None             # Baby's age in weeks
    feeding_type: Optional[str] = None            # "breast" or "formula"


# ═══════════════════════════════════════════════════════════════════════════════
#  BIO OUTPUT — probabilities for each cause
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BioResult:
    """What the bio model outputs.

    node_a: P(hungry) from bio context
    node_b: P(tired) from bio context
    node_c: {belly_pain: P, burping: P, discomfort: P} from bio context
    warnings: any clinical warnings (e.g. "possible fever")
    completeness: how much info the parent provided (0 to 1)
    """
    node_a_hungry: float              # P(hungry) from bio
    node_b_tired: float               # P(tired) from bio
    node_c: Dict[str, float]          # {belly_pain, burping, discomfort}
    warnings: List[str] = field(default_factory=list)
    completeness: float = 0.0         # How much context we have (0-1)

    def __repr__(self):
        return (f"BioResult(hungry={self.node_a_hungry:.3f}, "
                f"tired={self.node_b_tired:.3f}, "
                f"node_c={self.node_c}, "
                f"complete={self.completeness:.0%})")


# ═══════════════════════════════════════════════════════════════════════════════
#  AGE-BASED THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

def _get_age_profile(age_weeks: Optional[float]) -> Dict[str, float]:
    """Get age-appropriate thresholds for feeding and sleep.

    Based on pediatric guidelines:
        Neonate (0-4 weeks):    feeds every 1.5-2.5h, awake max 45min
        Young (4-12 weeks):     feeds every 2-3h, awake max 60min
        Infant (12-26 weeks):   feeds every 2.5-4h, awake max 90min
        Older (26-52 weeks):    feeds every 3-5h, awake max 120min

    Returns thresholds in MINUTES.
    """
    if age_weeks is None:
        age_weeks = 12  # Default: assume 3 months if unknown

    if age_weeks < 4:
        return {
            "feed_interval_typical": 120,    # 2 hours
            "feed_interval_hungry": 150,     # 2.5 hours → likely hungry
            "feed_interval_very_hungry": 180,# 3 hours → very hungry
            "max_awake": 45,                 # 45 min awake window
            "awake_tired": 50,               # likely tired after this
            "awake_very_tired": 70,          # very tired
            "diaper_interval": 90,           # check diaper every 1.5h
        }
    elif age_weeks < 12:
        return {
            "feed_interval_typical": 150,
            "feed_interval_hungry": 180,
            "feed_interval_very_hungry": 240,
            "max_awake": 60,
            "awake_tired": 75,
            "awake_very_tired": 100,
            "diaper_interval": 120,
        }
    elif age_weeks < 26:
        return {
            "feed_interval_typical": 210,
            "feed_interval_hungry": 240,
            "feed_interval_very_hungry": 300,
            "max_awake": 90,
            "awake_tired": 110,
            "awake_very_tired": 150,
            "diaper_interval": 150,
        }
    else:
        return {
            "feed_interval_typical": 240,
            "feed_interval_hungry": 300,
            "feed_interval_very_hungry": 360,
            "max_awake": 120,
            "awake_tired": 150,
            "awake_very_tired": 180,
            "diaper_interval": 180,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  SIGMOID HELPER — smooth probability transition
# ═══════════════════════════════════════════════════════════════════════════════

def _sigmoid(x: float, center: float, steepness: float = 0.05) -> float:
    """Smooth transition from 0 to 1.

    At x = center, returns 0.5.
    Steepness controls how sharp the transition is.
    Used to convert "minutes since feed" into P(hungry) smoothly.
    """
    z = steepness * (x - center)
    z = np.clip(z, -20, 20)  # Prevent overflow
    return float(1.0 / (1.0 + np.exp(-z)))


# ═══════════════════════════════════════════════════════════════════════════════
#  BIO MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class BioModel:
    """Rule-based biological context model.

    Converts parent-entered context into cause probabilities.
    No training needed — uses medical guidelines directly.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def predict(self, bio: BioInput) -> BioResult:
        """Convert bio context → cause probabilities.

        Each sub-model outputs a probability based on the relevant input.
        Missing inputs → return neutral prior (0.5 for binary, 0.33 for 3-class).
        """
        warnings = []
        n_provided = 0
        n_total = 7  # Total possible inputs

        profile = _get_age_profile(bio.age_weeks)
        if bio.age_weeks is not None:
            n_provided += 1

        # ── Node A: P(hungry) ─────────────────────────────────────────────
        if bio.minutes_since_feed is not None:
            n_provided += 1
            msf = bio.minutes_since_feed

            # Breastfed babies get hungry faster
            adjustment = 0.85 if bio.feeding_type == "breast" else 1.0
            if bio.feeding_type is not None:
                n_provided += 1

            # Center = typical feed interval (P=0.5 when at normal interval)
            # Beyond that → probability rises toward 1
            center = profile["feed_interval_typical"] * adjustment
            p_hungry = _sigmoid(msf, center, steepness=0.03)

            # Very long since feed → cap at high probability
            if msf > profile["feed_interval_very_hungry"] * adjustment:
                p_hungry = max(p_hungry, 0.85)
            # Just fed → very low
            if msf < 30:
                p_hungry = min(p_hungry, 0.10)
        else:
            p_hungry = 0.5  # No info → neutral

        # ── Node B: P(tired) ──────────────────────────────────────────────
        # Use minutes_awake if available, else minutes_since_sleep
        awake_time = None
        if bio.minutes_awake is not None:
            awake_time = bio.minutes_awake
            n_provided += 1
        elif bio.minutes_since_sleep is not None:
            awake_time = bio.minutes_since_sleep
            n_provided += 1

        if awake_time is not None:
            # Center = typical max awake time for this age
            center = profile["max_awake"]
            p_tired = _sigmoid(awake_time, center, steepness=0.035)

            if awake_time > profile["awake_very_tired"]:
                p_tired = max(p_tired, 0.85)
            if awake_time < 20:
                p_tired = min(p_tired, 0.10)
        else:
            p_tired = 0.5

        # ── Node C: P(belly_pain), P(burping), P(discomfort) ──────────────
        p_belly = 0.33
        p_burp = 0.33
        p_discomfort = 0.34

        # Temperature check
        if bio.temp_c is not None:
            n_provided += 1
            if bio.temp_c >= 38.0:
                warnings.append(f"Possible fever: {bio.temp_c:.1f}°C")
                p_discomfort += 0.20
                p_belly -= 0.05
                p_burp -= 0.05
            elif bio.temp_c >= 37.5:
                warnings.append(f"Slightly elevated temp: {bio.temp_c:.1f}°C")
                p_discomfort += 0.10
            elif bio.temp_c < 36.0:
                warnings.append(f"Low temperature: {bio.temp_c:.1f}°C")
                p_discomfort += 0.15

        # Diaper check
        if bio.minutes_since_diaper is not None:
            n_provided += 1
            if bio.minutes_since_diaper > profile["diaper_interval"]:
                p_discomfort += 0.15
                warnings.append("Diaper may need changing")

        # Recently fed → burping is more likely (gas after feeding)
        if bio.minutes_since_feed is not None:
            if bio.minutes_since_feed < 30:
                p_burp += 0.25
                p_belly += 0.10
            elif bio.minutes_since_feed < 60:
                p_burp += 0.15

        # Young babies (< 12 weeks) → higher colic/belly pain risk
        if bio.age_weeks is not None and bio.age_weeks < 12:
            p_belly += 0.10

        # Normalize Node C to sum to 1
        total_c = p_belly + p_burp + p_discomfort
        p_belly /= total_c
        p_burp /= total_c
        p_discomfort /= total_c

        # Completeness score
        completeness = n_provided / n_total

        result = BioResult(
            node_a_hungry=float(np.clip(p_hungry, 0.01, 0.99)),
            node_b_tired=float(np.clip(p_tired, 0.01, 0.99)),
            node_c={
                "belly_pain": float(p_belly),
                "burping": float(p_burp),
                "discomfort": float(p_discomfort),
            },
            warnings=warnings,
            completeness=completeness,
        )

        if self.verbose:
            print(f"  Bio prediction: {result}")
            if warnings:
                for w in warnings:
                    print(f"  ⚠️  {w}")

        return result


# ═══════════════════════════════════════════════════════════════════════════════
#  BIO SIMULATOR — generate realistic test data
# ═══════════════════════════════════════════════════════════════════════════════

class BioSimulator:
    """Generate simulated bio inputs for testing.

    Each cause has a typical bio profile:
        hungry:     long since feed, short awake time
        tired:      short since feed, long awake time
        belly_pain: recently fed (< 30min), young baby
        burping:    just fed (< 20min)
        discomfort: any pattern, possibly high temp or dirty diaper
    """

    def __init__(self, random_state: int = 42):
        self.rng = np.random.RandomState(random_state)

    def generate(self, cause: str, n: int = 1) -> List[BioInput]:
        """Generate n bio inputs for a given cause.

        Adds realistic noise — not every hungry baby was fed 3 hours ago.
        Some will have ambiguous profiles (this is intentional).
        """
        results = []
        for _ in range(n):
            bio = self._generate_one(cause)
            results.append(bio)
        return results

    def _generate_one(self, cause: str) -> BioInput:
        rng = self.rng

        # Base values (will be overridden per cause)
        age = rng.choice([2, 4, 8, 12, 20, 30])
        feed_type = rng.choice(["breast", "formula"])

        if cause == "hungry":
            # Long since feed
            msf = rng.uniform(150, 300)
            awake = rng.uniform(30, 90)
            diaper = rng.uniform(30, 120)
            temp = rng.normal(37.0, 0.2)

        elif cause == "tired":
            # Long awake time
            msf = rng.uniform(30, 120)
            awake = rng.uniform(80, 180)
            diaper = rng.uniform(30, 90)
            temp = rng.normal(37.0, 0.2)

        elif cause == "belly_pain":
            # Recently fed, young baby likely
            msf = rng.uniform(10, 60)
            awake = rng.uniform(20, 80)
            diaper = rng.uniform(20, 90)
            temp = rng.normal(37.1, 0.3)
            age = rng.choice([2, 4, 6, 8, 10])

        elif cause == "burping":
            # Just fed
            msf = rng.uniform(5, 25)
            awake = rng.uniform(15, 60)
            diaper = rng.uniform(20, 80)
            temp = rng.normal(37.0, 0.2)

        elif cause == "discomfort":
            # Could be anything — dirty diaper, temperature, etc.
            msf = rng.uniform(60, 180)
            awake = rng.uniform(30, 100)
            diaper = rng.uniform(90, 200)  # Long since diaper change
            temp = rng.normal(37.3, 0.5)   # Slightly elevated

        else:
            # Unknown cause — random
            msf = rng.uniform(30, 240)
            awake = rng.uniform(20, 120)
            diaper = rng.uniform(20, 150)
            temp = rng.normal(37.0, 0.3)

        # Add noise: sometimes parent doesn't fill in all fields
        if rng.random() < 0.15:
            msf = None
        if rng.random() < 0.15:
            awake = None
        if rng.random() < 0.20:
            diaper = None
        if rng.random() < 0.20:
            temp = None

        return BioInput(
            minutes_since_feed=msf,
            minutes_since_sleep=None,  # Use minutes_awake instead
            minutes_awake=awake,
            minutes_since_diaper=diaper,
            temp_c=float(np.clip(temp, 35.5, 40.0)) if temp is not None else None,
            age_weeks=float(age),
            feeding_type=feed_type,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Testing bio_model.py")
    print("=" * 60)

    model = BioModel(verbose=True)

    # ── Test 1: Hungry scenario ───────────────────────────────────────────
    print("\n  Test 1: Baby fed 3 hours ago, 8 weeks old")
    bio = BioInput(minutes_since_feed=180, age_weeks=8, feeding_type="formula",
                   minutes_awake=40, temp_c=37.0, minutes_since_diaper=60)
    r = model.predict(bio)
    assert r.node_a_hungry > 0.7, f"Expected hungry > 0.7, got {r.node_a_hungry}"
    print(f"  ✅ P(hungry) = {r.node_a_hungry:.3f} (high, correct)")

    # ── Test 2: Tired scenario ────────────────────────────────────────────
    print("\n  Test 2: Baby awake for 2 hours, just fed")
    bio = BioInput(minutes_since_feed=30, minutes_awake=120, age_weeks=12,
                   temp_c=37.0)
    r = model.predict(bio)
    assert r.node_b_tired > 0.7, f"Expected tired > 0.7, got {r.node_b_tired}"
    assert r.node_a_hungry < 0.3, f"Expected not hungry, got {r.node_a_hungry}"
    print(f"  ✅ P(tired) = {r.node_b_tired:.3f} (high)")
    print(f"  ✅ P(hungry) = {r.node_a_hungry:.3f} (low)")

    # ── Test 3: Just fed → burping likely ─────────────────────────────────
    print("\n  Test 3: Baby fed 10 minutes ago")
    bio = BioInput(minutes_since_feed=10, minutes_awake=30, age_weeks=6,
                   temp_c=37.0)
    r = model.predict(bio)
    assert r.node_c["burping"] > r.node_c["discomfort"], "Expected burping > discomfort"
    print(f"  ✅ Node C: burping={r.node_c['burping']:.3f} > "
          f"discomfort={r.node_c['discomfort']:.3f}")

    # ── Test 4: Fever ─────────────────────────────────────────────────────
    print("\n  Test 4: Baby has fever (38.5°C)")
    bio = BioInput(temp_c=38.5, age_weeks=10)
    r = model.predict(bio)
    assert "fever" in r.warnings[0].lower(), f"Expected fever warning"
    assert r.node_c["discomfort"] > 0.40, "Expected discomfort high"
    print(f"  ✅ Warning: {r.warnings[0]}")
    print(f"  ✅ Discomfort = {r.node_c['discomfort']:.3f} (elevated)")

    # ── Test 5: Missing data → neutral ────────────────────────────────────
    print("\n  Test 5: No information provided")
    bio = BioInput()
    r = model.predict(bio)
    assert 0.4 <= r.node_a_hungry <= 0.6, "Expected neutral with no info"
    assert 0.4 <= r.node_b_tired <= 0.6, "Expected neutral with no info"
    assert r.completeness == 0.0
    print(f"  ✅ P(hungry) = {r.node_a_hungry:.3f} (neutral)")
    print(f"  ✅ P(tired) = {r.node_b_tired:.3f} (neutral)")
    print(f"  ✅ Completeness = {r.completeness:.0%} (no data)")

    # ── Test 6: Simulator ─────────────────────────────────────────────────
    print("\n  Test 6: Bio simulator")
    sim = BioSimulator(random_state=42)
    for cause in ["hungry", "tired", "belly_pain", "burping", "discomfort"]:
        bios = sim.generate(cause, n=3)
        r = model.predict(bios[0])
        print(f"  {cause:<14} → hungry={r.node_a_hungry:.2f}  "
              f"tired={r.node_b_tired:.2f}  "
              f"belly={r.node_c['belly_pain']:.2f}  "
              f"burp={r.node_c['burping']:.2f}  "
              f"disc={r.node_c['discomfort']:.2f}")

    print(f"\n{'=' * 60}")
    print(f"  bio_model.py — ALL TESTS PASSED ✅")
    print(f"{'=' * 60}")
