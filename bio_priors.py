"""
bio_priors.py — Age-adaptive prior probabilities for cry causes.

This file contains STATISTICAL PRIORS — the baseline probability of each
cause BEFORE seeing any audio or context data.

Why priors matter:
    A 2-week-old who is crying has different base rates than a 6-month-old:
    - Neonates: very frequent hunger (every 1.5h), colic peaks at 3-12 weeks
    - Older babies: longer feed intervals, less colic, more behavioral crying

    These priors come from pediatric literature, not from our dataset.
    They represent "what's the most likely cause for a baby this age?"

How they're used:
    1. bio_model.py combines priors + context (time since feed, etc.)
    2. fusion module uses priors when audio is uncertain
    3. If NO context is available, priors alone give a reasonable estimate

Age stages (from pediatric guidelines):
    Neonate:      0–4 weeks    (very frequent feeding, colic rare)
    Young infant: 4–12 weeks   (peak colic period, growth spurts)
    Infant:       12–26 weeks  (longer sleep cycles, more predictable)
    Older infant: 26–52 weeks  (solid foods starting, different patterns)
"""

from dataclasses import dataclass
from typing import Dict, Optional


# ═══════════════════════════════════════════════════════════════════════════════
#  AGE PROFILE — feeding/sleep thresholds per developmental stage
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AgeProfile:
    """Age-specific thresholds in MINUTES.

    Sources:
        - AAP (American Academy of Pediatrics) feeding guidelines
        - Weissbluth "Healthy Sleep Habits, Happy Child"
        - Hiscock et al. (2014) infant sleep patterns meta-analysis
    """
    stage_name: str

    # Feeding thresholds
    feed_typical: float         # Normal interval between feeds
    feed_hungry: float          # Baby probably hungry after this long
    feed_very_hungry: float     # Baby almost certainly hungry

    # Wake window thresholds
    max_awake: float            # Typical max awake time
    awake_tired: float          # Baby probably tired after this
    awake_overtired: float      # Baby is overtired (harder to settle)

    # Other
    diaper_interval: float      # Typical diaper change frequency
    colic_risk: float           # Relative colic risk (0 to 1)


# ── The four developmental stages ─────────────────────────────────────────────

AGE_PROFILES = {
    "neonate": AgeProfile(
        stage_name="neonate",
        feed_typical=120,        # every 2 hours
        feed_hungry=150,         # 2.5 hours
        feed_very_hungry=180,    # 3 hours
        max_awake=45,
        awake_tired=50,
        awake_overtired=70,
        diaper_interval=90,
        colic_risk=0.3,
    ),
    "young_infant": AgeProfile(
        stage_name="young_infant",
        feed_typical=150,        # every 2.5 hours
        feed_hungry=180,         # 3 hours
        feed_very_hungry=240,    # 4 hours
        max_awake=60,
        awake_tired=75,
        awake_overtired=100,
        diaper_interval=120,
        colic_risk=1.0,          # PEAK colic period (3–12 weeks)
    ),
    "infant": AgeProfile(
        stage_name="infant",
        feed_typical=210,        # every 3.5 hours
        feed_hungry=240,         # 4 hours
        feed_very_hungry=300,    # 5 hours
        max_awake=90,
        awake_tired=110,
        awake_overtired=150,
        diaper_interval=150,
        colic_risk=0.4,
    ),
    "older_infant": AgeProfile(
        stage_name="older_infant",
        feed_typical=240,        # every 4 hours
        feed_hungry=300,         # 5 hours
        feed_very_hungry=360,    # 6 hours
        max_awake=120,
        awake_tired=150,
        awake_overtired=180,
        diaper_interval=180,
        colic_risk=0.15,
    ),
}


def get_age_profile(age_weeks: Optional[float]) -> AgeProfile:
    """Get the appropriate age profile.

    Returns the profile for the baby's developmental stage.
    If age is unknown, defaults to young_infant (most conservative).
    """
    if age_weeks is None:
        return AGE_PROFILES["young_infant"]

    if age_weeks < 4:
        return AGE_PROFILES["neonate"]
    elif age_weeks < 12:
        return AGE_PROFILES["young_infant"]
    elif age_weeks < 26:
        return AGE_PROFILES["infant"]
    else:
        return AGE_PROFILES["older_infant"]


# ═══════════════════════════════════════════════════════════════════════════════
#  BASE RATE PRIORS — "before seeing any data, what's most likely?"
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CausePriors:
    """Baseline probability of each cause for a given age.

    These represent: "A baby of this age is crying. What's the
    most likely cause BEFORE we know anything else?"

    Sources:
        - Barr et al. (2005) "Crying patterns in infancy"
        - Wolke et al. (2017) "Systematic review of crying"
        - St James-Roberts (2012) "The Origins, Prevention and
          Treatment of Infant Crying and Sleeping Problems"
    """
    hungry: float
    tired: float
    belly_pain: float
    burping: float
    discomfort: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "hungry": self.hungry,
            "tired": self.tired,
            "belly_pain": self.belly_pain,
            "burping": self.burping,
            "discomfort": self.discomfort,
        }


# Base rates change with age:
#   - Neonates cry mostly from hunger (they feed constantly)
#   - Young infants have peak colic (belly pain spikes)
#   - Older infants have more varied causes

CAUSE_PRIORS = {
    "neonate": CausePriors(
        hungry=0.40,         # Dominant cause at this age
        tired=0.20,
        belly_pain=0.10,
        burping=0.15,        # Common after frequent feeds
        discomfort=0.15,
    ),
    "young_infant": CausePriors(
        hungry=0.30,
        tired=0.20,
        belly_pain=0.20,     # Colic peaks here
        burping=0.15,
        discomfort=0.15,
    ),
    "infant": CausePriors(
        hungry=0.25,
        tired=0.25,          # More predictable sleep needs
        belly_pain=0.15,
        burping=0.10,
        discomfort=0.25,     # More environmental awareness
    ),
    "older_infant": CausePriors(
        hungry=0.20,
        tired=0.25,
        belly_pain=0.10,
        burping=0.10,
        discomfort=0.35,     # Teething, frustration, etc.
    ),
}


def get_cause_priors(age_weeks: Optional[float]) -> CausePriors:
    """Get baseline cause probabilities for a given age."""
    if age_weeks is None:
        return CAUSE_PRIORS["young_infant"]

    if age_weeks < 4:
        return CAUSE_PRIORS["neonate"]
    elif age_weeks < 12:
        return CAUSE_PRIORS["young_infant"]
    elif age_weeks < 26:
        return CAUSE_PRIORS["infant"]
    else:
        return CAUSE_PRIORS["older_infant"]


# ═══════════════════════════════════════════════════════════════════════════════
#  FEEDING TYPE ADJUSTMENT
# ═══════════════════════════════════════════════════════════════════════════════

def get_feed_adjustment(feeding_type: Optional[str]) -> float:
    """Breastfed babies digest faster → get hungry sooner.

    Returns a multiplier for feed interval thresholds:
        breast:  0.85 (15% shorter intervals → hungry faster)
        formula: 1.00 (baseline)
        unknown: 0.92 (slight adjustment toward shorter)
    """
    if feeding_type == "breast":
        return 0.85
    elif feeding_type == "formula":
        return 1.00
    else:
        return 0.92


# ═══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing bio_priors.py ...\n")

    # Test age profiles
    for age in [1, 6, 16, 30, None]:
        p = get_age_profile(age)
        print(f"  Age {str(age) + 'w':>5} → {p.stage_name:<14} "
              f"feed_hungry={p.feed_hungry}min  "
              f"max_awake={p.max_awake}min  colic={p.colic_risk}")

    # Test priors
    print()
    for age in [1, 6, 16, 30]:
        priors = get_cause_priors(age)
        d = priors.as_dict()
        total = sum(d.values())
        print(f"  Age {age:2d}w priors: {d}  (sum={total:.2f})")
        assert abs(total - 1.0) < 0.01, f"Priors don't sum to 1: {total}"

    # Test feed adjustment
    assert get_feed_adjustment("breast") < get_feed_adjustment("formula")
    print(f"\n  Feed adjustment breast={get_feed_adjustment('breast'):.2f}  "
          f"formula={get_feed_adjustment('formula'):.2f}")

    print("\n  bio_priors.py — OK ✅")