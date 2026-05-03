"""
clinical_rules.py — Medical safety rules and clinical warnings.

This file contains EXPERT MEDICAL KNOWLEDGE encoded as rules.
These are NOT learned from data — they come from pediatric guidelines.

Separation from bio_priors.py:
    bio_priors    = "what's statistically likely for this age"
    clinical_rules = "what medical red flags should we check"

Clinical rules serve three purposes:
    1. WARNINGS: alert parent to potential medical issues (fever, dehydration)
    2. ADJUSTMENTS: shift probabilities based on clinical signs
    3. URGENCY: flag cases that need a doctor, not just a diaper change

Sources:
    - WHO/UNICEF infant care guidelines
    - AAP "Bright Futures" pediatric guidelines
    - NICE CG149 "Fever in under 5s"
    - Hyman et al. (2006) "Childhood functional GI disorders"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  WARNING LEVELS
# ═══════════════════════════════════════════════════════════════════════════════

class Urgency:
    """How urgent is this warning?"""
    INFO = "info"           # Good to know, not urgent
    CAUTION = "caution"     # Parent should monitor
    WARNING = "warning"     # Should act soon
    URGENT = "urgent"       # See a doctor


@dataclass
class ClinicalWarning:
    """A single clinical warning."""
    message: str
    urgency: str            # One of Urgency levels
    rule_name: str          # Which rule triggered this
    adjustment: Dict[str, float] = field(default_factory=dict)
    # adjustment: {"discomfort": +0.20} means add 0.20 to discomfort probability


# ═══════════════════════════════════════════════════════════════════════════════
#  TEMPERATURE RULES
# ═══════════════════════════════════════════════════════════════════════════════

def check_temperature(
    temp_c: Optional[float],
    age_weeks: Optional[float] = None,
) -> List[ClinicalWarning]:
    """Check baby's temperature against clinical thresholds.

    Pediatric fever definitions:
        Normal:    36.5°C – 37.5°C
        Low-grade: 37.5°C – 38.0°C
        Fever:     38.0°C – 39.0°C
        High fever: > 39.0°C

    Special rule: ANY fever in a neonate (< 4 weeks) is URGENT.
    Neonates can't regulate temperature well and fever may indicate
    serious infection.
    """
    if temp_c is None:
        return []

    warnings = []
    age = age_weeks or 12  # Default if unknown

    # ── Hypothermia ───────────────────────────────────────────────────────
    if temp_c < 35.5:
        warnings.append(ClinicalWarning(
            message=f"Hypothermia: {temp_c:.1f}°C — seek medical attention",
            urgency=Urgency.URGENT,
            rule_name="hypothermia",
            adjustment={"discomfort": 0.30},
        ))
    elif temp_c < 36.0:
        warnings.append(ClinicalWarning(
            message=f"Low temperature: {temp_c:.1f}°C — keep baby warm",
            urgency=Urgency.WARNING,
            rule_name="low_temp",
            adjustment={"discomfort": 0.15},
        ))

    # ── Fever ─────────────────────────────────────────────────────────────
    if temp_c >= 39.0:
        warnings.append(ClinicalWarning(
            message=f"High fever: {temp_c:.1f}°C — see a doctor",
            urgency=Urgency.URGENT,
            rule_name="high_fever",
            adjustment={"discomfort": 0.30, "belly_pain": -0.05},
        ))
    elif temp_c >= 38.0:
        # Neonate with fever = always urgent
        if age < 4:
            warnings.append(ClinicalWarning(
                message=f"Fever in neonate: {temp_c:.1f}°C — see a doctor immediately",
                urgency=Urgency.URGENT,
                rule_name="neonate_fever",
                adjustment={"discomfort": 0.30},
            ))
        else:
            warnings.append(ClinicalWarning(
                message=f"Fever: {temp_c:.1f}°C — monitor closely",
                urgency=Urgency.WARNING,
                rule_name="fever",
                adjustment={"discomfort": 0.20},
            ))
    elif temp_c >= 37.5:
        warnings.append(ClinicalWarning(
            message=f"Slightly elevated temperature: {temp_c:.1f}°C",
            urgency=Urgency.CAUTION,
            rule_name="elevated_temp",
            adjustment={"discomfort": 0.10},
        ))

    return warnings


# ═══════════════════════════════════════════════════════════════════════════════
#  FEEDING PATTERN RULES
# ═══════════════════════════════════════════════════════════════════════════════

def check_feeding_pattern(
    minutes_since_feed: Optional[float],
    age_weeks: Optional[float] = None,
    feed_hungry_threshold: float = 180,
) -> List[ClinicalWarning]:
    """Check for concerning feeding patterns.

    Red flags:
        - Very long since last feed (> 2× typical interval): dehydration risk
        - Neonate not fed in > 4 hours: medical concern
    """
    if minutes_since_feed is None:
        return []

    warnings = []
    age = age_weeks or 12

    # Neonate not fed in > 4 hours
    if age < 4 and minutes_since_feed > 240:
        warnings.append(ClinicalWarning(
            message=f"Neonate not fed in {minutes_since_feed:.0f} min — ensure feeding",
            urgency=Urgency.WARNING,
            rule_name="neonate_feed_gap",
            adjustment={"hungry": 0.15},
        ))

    # Any baby not fed in > 6 hours
    elif minutes_since_feed > 360:
        warnings.append(ClinicalWarning(
            message=f"Long feed gap: {minutes_since_feed:.0f} min — check hydration",
            urgency=Urgency.CAUTION,
            rule_name="long_feed_gap",
            adjustment={"hungry": 0.10},
        ))

    return warnings


# ═══════════════════════════════════════════════════════════════════════════════
#  DIAPER RULES
# ═══════════════════════════════════════════════════════════════════════════════

def check_diaper(
    minutes_since_diaper: Optional[float],
    diaper_threshold: float = 150,
) -> List[ClinicalWarning]:
    """Check if diaper needs changing."""
    if minutes_since_diaper is None:
        return []

    warnings = []

    if minutes_since_diaper > diaper_threshold * 1.5:
        warnings.append(ClinicalWarning(
            message=f"Diaper not changed in {minutes_since_diaper:.0f} min",
            urgency=Urgency.CAUTION,
            rule_name="diaper_overdue",
            adjustment={"discomfort": 0.15},
        ))
    elif minutes_since_diaper > diaper_threshold:
        warnings.append(ClinicalWarning(
            message="Diaper may need changing",
            urgency=Urgency.INFO,
            rule_name="diaper_due",
            adjustment={"discomfort": 0.08},
        ))

    return warnings


# ═══════════════════════════════════════════════════════════════════════════════
#  SLEEP PATTERN RULES
# ═══════════════════════════════════════════════════════════════════════════════

def check_sleep_pattern(
    minutes_awake: Optional[float],
    awake_overtired: float = 100,
) -> List[ClinicalWarning]:
    """Check for overtiredness.

    An overtired baby is harder to settle and may cry more intensely.
    This doesn't mean tiredness is the cause — but it affects all causes.
    """
    if minutes_awake is None:
        return []

    warnings = []

    if minutes_awake > awake_overtired * 1.5:
        warnings.append(ClinicalWarning(
            message=f"Baby has been awake {minutes_awake:.0f} min — very overtired",
            urgency=Urgency.CAUTION,
            rule_name="very_overtired",
            adjustment={"tired": 0.15},
        ))
    elif minutes_awake > awake_overtired:
        warnings.append(ClinicalWarning(
            message=f"Baby may be overtired (awake {minutes_awake:.0f} min)",
            urgency=Urgency.INFO,
            rule_name="overtired",
            adjustment={"tired": 0.08},
        ))

    return warnings


# ═══════════════════════════════════════════════════════════════════════════════
#  POST-FEED RULES (for Node C)
# ═══════════════════════════════════════════════════════════════════════════════

def check_post_feed(
    minutes_since_feed: Optional[float],
    age_weeks: Optional[float] = None,
) -> List[ClinicalWarning]:
    """Check if crying is post-feed related (burping/gas).

    Within 30 minutes of feeding:
        - Burping is very likely
        - Belly pain from gas is possible
        - Hunger is very unlikely

    Colic risk increases with:
        - Age 3-12 weeks (peak colic period)
        - Recent feeding (gas)
    """
    if minutes_since_feed is None:
        return []

    warnings = []
    age = age_weeks or 12

    if minutes_since_feed < 20:
        warnings.append(ClinicalWarning(
            message="Just fed — likely needs burping",
            urgency=Urgency.INFO,
            rule_name="post_feed_burp",
            adjustment={"burping": 0.25, "belly_pain": 0.10, "hungry": -0.15},
        ))
    elif minutes_since_feed < 45:
        warnings.append(ClinicalWarning(
            message="Recently fed — gas or burping possible",
            urgency=Urgency.INFO,
            rule_name="recent_feed_gas",
            adjustment={"burping": 0.15, "belly_pain": 0.08},
        ))

    # Colic risk for young infants
    if age is not None and 3 <= age <= 12 and minutes_since_feed is not None:
        if minutes_since_feed < 60:
            warnings.append(ClinicalWarning(
                message="Peak colic age — consider gas/colic",
                urgency=Urgency.INFO,
                rule_name="colic_age",
                adjustment={"belly_pain": 0.10},
            ))

    return warnings


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN ALL RULES
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_checks(
    temp_c: Optional[float] = None,
    minutes_since_feed: Optional[float] = None,
    minutes_awake: Optional[float] = None,
    minutes_since_diaper: Optional[float] = None,
    age_weeks: Optional[float] = None,
    diaper_threshold: float = 150,
    feed_hungry_threshold: float = 180,
    awake_overtired: float = 100,
) -> List[ClinicalWarning]:
    """Run all clinical checks and collect warnings.

    Returns a list of ClinicalWarning objects, sorted by urgency.
    """
    all_warnings = []

    all_warnings.extend(check_temperature(temp_c, age_weeks))
    all_warnings.extend(check_feeding_pattern(minutes_since_feed, age_weeks,
                                               feed_hungry_threshold))
    all_warnings.extend(check_diaper(minutes_since_diaper, diaper_threshold))
    all_warnings.extend(check_sleep_pattern(minutes_awake, awake_overtired))
    all_warnings.extend(check_post_feed(minutes_since_feed, age_weeks))

    # Sort by urgency (urgent first)
    urgency_order = {
        Urgency.URGENT: 0,
        Urgency.WARNING: 1,
        Urgency.CAUTION: 2,
        Urgency.INFO: 3,
    }
    all_warnings.sort(key=lambda w: urgency_order.get(w.urgency, 4))

    return all_warnings


def aggregate_adjustments(warnings: List[ClinicalWarning]) -> Dict[str, float]:
    """Sum up all probability adjustments from clinical rules.

    Returns dict like {"hungry": +0.15, "discomfort": +0.30, "burping": +0.25}
    """
    combined = {}
    for w in warnings:
        for cause, delta in w.adjustment.items():
            combined[cause] = combined.get(cause, 0.0) + delta
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing clinical_rules.py ...\n")

    # Test 1: Fever
    w = check_temperature(38.5, age_weeks=10)
    assert len(w) > 0
    assert w[0].urgency == Urgency.WARNING
    print(f"  ✅ Fever 38.5°C: {w[0].message} [{w[0].urgency}]")

    # Test 2: Neonate fever → URGENT
    w = check_temperature(38.2, age_weeks=2)
    assert w[0].urgency == Urgency.URGENT
    print(f"  ✅ Neonate fever: {w[0].message} [{w[0].urgency}]")

    # Test 3: High fever
    w = check_temperature(39.5)
    assert w[0].urgency == Urgency.URGENT
    print(f"  ✅ High fever: {w[0].message} [{w[0].urgency}]")

    # Test 4: Hypothermia
    w = check_temperature(35.0)
    assert w[0].urgency == Urgency.URGENT
    print(f"  ✅ Hypothermia: {w[0].message} [{w[0].urgency}]")

    # Test 5: Normal temp → no warnings
    w = check_temperature(37.0)
    assert len(w) == 0
    print(f"  ✅ Normal temp: no warnings")

    # Test 6: Post-feed burping
    w = check_post_feed(10, age_weeks=8)
    assert any("burp" in x.message.lower() for x in w)
    print(f"  ✅ Just fed: {w[0].message}")

    # Test 7: Colic age
    w = check_post_feed(30, age_weeks=6)
    assert any("colic" in x.message.lower() for x in w)
    print(f"  ✅ Colic age: {[x.message for x in w]}")

    # Test 8: Diaper overdue
    w = check_diaper(250, diaper_threshold=150)
    assert len(w) > 0
    print(f"  ✅ Diaper: {w[0].message}")

    # Test 9: Overtired
    w = check_sleep_pattern(160, awake_overtired=100)
    assert len(w) > 0
    print(f"  ✅ Overtired: {w[0].message}")

    # Test 10: Run all checks
    all_w = run_all_checks(
        temp_c=38.5, minutes_since_feed=15,
        minutes_awake=130, minutes_since_diaper=200,
        age_weeks=6,
    )
    print(f"\n  All checks (fever + just fed + overtired + dirty diaper):")
    for w in all_w:
        print(f"    [{w.urgency:>8}] {w.message}")

    # Test 11: Aggregate adjustments
    adj = aggregate_adjustments(all_w)
    print(f"\n  Combined adjustments: {adj}")

    print(f"\n  clinical_rules.py — OK ✅")