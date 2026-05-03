"""
fusion.py — Multimodal Fusion Engine.

THE CORE OF THE SYSTEM.

Takes predictions from 3 modalities (audio, bio, video) and combines
them into a final diagnosis. This is where Node A's weak AUC=0.56
becomes useful when combined with bio context.

FUSION METHOD: Reliability-Weighted Product of Experts (RW-PoE)

    P(cause | all modalities) ∝ ∏_m  P_m(cause)^(w_m)

    where w_m = reliability of modality m:
        - Audio:  based on model confidence + cry gate quality
        - Bio:    based on how much context parent provided
        - Video:  0.0 (missing), 0.3 (simulated), 0.7 (real)

    Missing modality → w_m = 0 → P_m^0 = 1 → no effect on fusion.
    This is the mathematically correct way to "ignore" a modality.

WHY RW-PoE AND NOT SIMPLE AVERAGING:
    Average:  P_fused = (0.56 + 0.71 + 0.50) / 3 = 0.59
    RW-PoE:   P_fused = 0.56^0.4 × 0.71^0.5 × 0.50^0.0 = higher for hungry
              because bio (reliability=0.5) gets more weight than audio (0.4)
              and missing video (0.0) is properly ignored.

HANDLES ANY COMBINATION:
    Audio + Bio + Video  → full fusion
    Audio + Bio          → video weight = 0, ignored
    Audio only           → bio and video weights = 0
    Bio only             → audio and video weights = 0
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

EPS = 1e-12

# All possible final causes the system can output
ALL_CAUSES = ["hungry", "tired", "belly_pain", "burping", "discomfort"]


# ═══════════════════════════════════════════════════════════════════════════════
#  INPUT: what each modality provides to fusion
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModalityInput:
    """One modality's prediction for one node.

    p_positive: probability of the positive class (for binary nodes)
                OR dict of probabilities (for 3-class Node C)
    reliability: how much to trust this modality (0 to 1)
    source: name of the modality ("audio", "bio", "video")
    """
    source: str
    reliability: float
    # For binary nodes (A, B)
    p_positive: Optional[float] = None
    # For multi-class node (C)
    p_classes: Optional[Dict[str, float]] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT: what fusion produces
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NodeFusionResult:
    """Fused result for a single node."""
    node: str                          # "node_a", "node_b", "node_c"
    posterior: Dict[str, float]        # Fused probabilities
    prediction: str                    # Most likely class
    confidence: float                  # P(prediction)
    conflict_score: float              # How much modalities disagree (0-1)
    modalities_used: List[str]         # Which modalities contributed
    is_uncertain: bool                 # True if confidence < threshold


@dataclass
class FinalDiagnosis:
    """The system's final output — what the app shows to the parent.

    top_cause:    most likely cause ("hungry", "tired", etc.)
    all_proba:    probability for every cause
    confidence:   how sure we are (0 to 1)
    is_uncertain: should we say "I'm not sure"?
    conflicts:    any modality disagreements
    warnings:     clinical warnings from bio model
    node_results: detailed per-node fusion results
    """
    top_cause: str
    all_proba: Dict[str, float]
    confidence: float
    is_uncertain: bool
    conflicts: List[str]
    warnings: List[str]
    node_results: Dict[str, NodeFusionResult]
    modalities_available: List[str]

    def __repr__(self):
        mods = "+".join(self.modalities_available)
        return (f"Diagnosis({self.top_cause}, conf={self.confidence:.3f}, "
                f"uncertain={self.is_uncertain}, mods=[{mods}])")

    def summary(self) -> str:
        """Human-readable summary for the app."""
        lines = []
        lines.append(f"  Prediction: {self.top_cause.upper()}")
        lines.append(f"  Confidence: {self.confidence:.0%}")
        if self.is_uncertain:
            lines.append("  ⚠️  Low confidence — multiple causes possible")
        lines.append(f"  Modalities: {', '.join(self.modalities_available)}")
        lines.append(f"\n  All probabilities:")
        for cause, p in sorted(self.all_proba.items(), key=lambda x: -x[1]):
            bar = "█" * int(p * 30)
            lines.append(f"    {cause:<14} {p:.1%}  {bar}")
        if self.conflicts:
            lines.append(f"\n  Conflicts:")
            for c in self.conflicts:
                lines.append(f"    ⚠️  {c}")
        if self.warnings:
            lines.append(f"\n  Clinical warnings:")
            for w in self.warnings:
                lines.append(f"    {w}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE: Reliability-Weighted Product of Experts
# ═══════════════════════════════════════════════════════════════════════════════

def _rw_poe_binary(inputs: List[ModalityInput]) -> Tuple[float, float, List[str]]:
    """Fuse binary predictions using Reliability-Weighted PoE.

    For each modality m with P_m(positive) and reliability w_m:
        log P(pos|all) ∝ Σ_m  w_m · log P_m(pos)
        log P(neg|all) ∝ Σ_m  w_m · log P_m(neg)

    If w_m = 0, the modality contributes nothing (log term = 0).

    Returns: (p_positive_fused, conflict_score, modalities_used)
    """
    log_pos = 0.0
    log_neg = 0.0
    used = []

    for inp in inputs:
        w = inp.reliability
        if w < 0.01 or inp.p_positive is None:
            continue  # Skip missing/unreliable modality

        p = np.clip(inp.p_positive, EPS, 1.0 - EPS)
        log_pos += w * np.log(p)
        log_neg += w * np.log(1.0 - p)
        used.append(inp.source)

    if not used:
        # No modalities available → return uniform
        return 0.5, 0.0, []

    # Normalize
    log_max = max(log_pos, log_neg)
    p_pos = np.exp(log_pos - log_max)
    p_neg = np.exp(log_neg - log_max)
    total = p_pos + p_neg + EPS
    p_fused = float(p_pos / total)

    # Conflict: how much do modalities disagree?
    conflict = _compute_conflict_binary(inputs)

    return p_fused, conflict, used


def _rw_poe_multiclass(
    inputs: List[ModalityInput],
    classes: List[str],
) -> Tuple[Dict[str, float], float, List[str]]:
    """Fuse multi-class predictions using RW-PoE.

    Same principle: log P(c|all) ∝ Σ_m  w_m · log P_m(c)

    Returns: (posterior_dict, conflict_score, modalities_used)
    """
    K = len(classes)
    log_post = np.zeros(K)
    used = []

    for inp in inputs:
        w = inp.reliability
        if w < 0.01 or inp.p_classes is None:
            continue

        for i, cls in enumerate(classes):
            p = np.clip(inp.p_classes.get(cls, 1.0 / K), EPS, 1.0 - EPS)
            log_post[i] += w * np.log(p)
        used.append(inp.source)

    if not used:
        return {c: 1.0 / K for c in classes}, 0.0, []

    # Normalize
    log_post -= log_post.max()
    post = np.exp(log_post)
    post = post / (post.sum() + EPS)

    posterior = {cls: float(post[i]) for i, cls in enumerate(classes)}

    conflict = _compute_conflict_multiclass(inputs, classes)

    return posterior, conflict, used


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFLICT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_conflict_binary(inputs: List[ModalityInput]) -> float:
    """Measure how much modalities disagree (binary case).

    Conflict = max pairwise disagreement between active modalities.
    Two modalities disagree if one says P>0.6 and the other says P<0.4.
    """
    active = [inp for inp in inputs if inp.reliability > 0.01 and inp.p_positive is not None]
    if len(active) < 2:
        return 0.0

    max_conflict = 0.0
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            diff = abs(active[i].p_positive - active[j].p_positive)
            max_conflict = max(max_conflict, diff)

    return float(max_conflict)


def _compute_conflict_multiclass(
    inputs: List[ModalityInput],
    classes: List[str],
) -> float:
    """Measure conflict for multi-class predictions.

    Conflict = do modalities pick DIFFERENT top classes?
    """
    active = [inp for inp in inputs if inp.reliability > 0.01 and inp.p_classes is not None]
    if len(active) < 2:
        return 0.0

    top_classes = []
    for inp in active:
        top = max(inp.p_classes, key=inp.p_classes.get)
        top_classes.append(top)

    # If all agree → conflict = 0, if all disagree → conflict = 1
    n_unique = len(set(top_classes))
    return float((n_unique - 1) / max(len(top_classes) - 1, 1))


# ═══════════════════════════════════════════════════════════════════════════════
#  FUSION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class FusionEngine:
    """Combines audio + bio + video predictions into a final diagnosis.

    Usage:
        engine = FusionEngine()
        diagnosis = engine.fuse(
            audio_nodeA=..., audio_nodeB=..., audio_nodeC=...,
            bio_result=...,
            video_result=...,     # None if no video
            cry_gate_result=...,  # None if no cry gate
        )
        print(diagnosis.summary())
    """

    def __init__(
        self,
        confidence_threshold: float = 0.40,
        conflict_threshold: float = 0.40,
        verbose: bool = True,
    ):
        self.conf_threshold = confidence_threshold
        self.conflict_threshold = conflict_threshold
        self.verbose = verbose

    def fuse(
        self,
        # Audio predictions (from trained models)
        audio_nodeA_p_hungry: Optional[float] = None,
        audio_nodeA_confidence: float = 0.5,
        audio_nodeB_p_tired: Optional[float] = None,
        audio_nodeB_confidence: float = 0.5,
        audio_nodeC_proba: Optional[Dict[str, float]] = None,
        audio_nodeC_confidence: float = 0.33,
        # Bio predictions (from parent context)
        bio_p_hungry: Optional[float] = None,
        bio_p_tired: Optional[float] = None,
        bio_node_c: Optional[Dict[str, float]] = None,
        bio_completeness: float = 0.0,
        # Video predictions (simulated or real)
        video_p_hungry: Optional[float] = None,
        video_p_tired: Optional[float] = None,
        video_node_c: Optional[Dict[str, float]] = None,
        video_reliability: float = 0.0,
        # Cry gate
        cry_gate_p: float = 1.0,
        cry_gate_quality: float = 1.0,
        # Clinical warnings from bio model
        clinical_warnings: Optional[List[str]] = None,
    ) -> FinalDiagnosis:
        """Run full fusion pipeline.

        Steps:
            1. Compute reliability weights for each modality
            2. Fuse Node A (hungry vs non-hungry)
            3. Fuse Node B (tired vs active)
            4. Fuse Node C (belly_pain / burping / discomfort)
            5. Combine nodes into final cause probabilities
            6. Detect conflicts and uncertainty
            7. Return FinalDiagnosis
        """

        # ── Step 1: Compute reliability weights ──────────────────────────
        # Audio reliability = model confidence × cry gate quality
        # Bio reliability = completeness (how much parent filled in)
        # Video reliability = from video model (0/0.3/0.7)

        w_audio_a = audio_nodeA_confidence * cry_gate_quality if audio_nodeA_p_hungry is not None else 0.0
        w_audio_b = audio_nodeB_confidence * cry_gate_quality if audio_nodeB_p_tired is not None else 0.0
        w_audio_c = audio_nodeC_confidence * cry_gate_quality if audio_nodeC_proba is not None else 0.0
        w_bio = bio_completeness if bio_p_hungry is not None else 0.0
        w_video = video_reliability

        available = []
        if max(w_audio_a, w_audio_b, w_audio_c) > 0.01:
            available.append("audio")
        if w_bio > 0.01:
            available.append("bio")
        if w_video > 0.01:
            available.append("video")

        if self.verbose:
            print(f"  Fusion weights: audio=[{w_audio_a:.2f},{w_audio_b:.2f},{w_audio_c:.2f}]  "
                  f"bio={w_bio:.2f}  video={w_video:.2f}")

        # ── Step 2: Fuse Node A (hungry vs non-hungry) ───────────────────
        inputs_a = [
            ModalityInput("audio", w_audio_a, p_positive=audio_nodeA_p_hungry),
            ModalityInput("bio", w_bio, p_positive=bio_p_hungry),
            ModalityInput("video", w_video, p_positive=video_p_hungry),
        ]
        p_hungry, conflict_a, used_a = _rw_poe_binary(inputs_a)

        node_a = NodeFusionResult(
            node="node_a",
            posterior={"hungry": p_hungry, "non_hungry": 1 - p_hungry},
            prediction="hungry" if p_hungry >= 0.5 else "non_hungry",
            confidence=max(p_hungry, 1 - p_hungry),
            conflict_score=conflict_a,
            modalities_used=used_a,
            is_uncertain=max(p_hungry, 1 - p_hungry) < self.conf_threshold + 0.10,
        )

        # ── Step 3: Fuse Node B (tired vs active) ────────────────────────
        inputs_b = [
            ModalityInput("audio", w_audio_b, p_positive=audio_nodeB_p_tired),
            ModalityInput("bio", w_bio, p_positive=bio_p_tired),
            ModalityInput("video", w_video, p_positive=video_p_tired),
        ]
        p_tired, conflict_b, used_b = _rw_poe_binary(inputs_b)

        node_b = NodeFusionResult(
            node="node_b",
            posterior={"tired": p_tired, "active": 1 - p_tired},
            prediction="tired" if p_tired >= 0.5 else "active",
            confidence=max(p_tired, 1 - p_tired),
            conflict_score=conflict_b,
            modalities_used=used_b,
            is_uncertain=max(p_tired, 1 - p_tired) < self.conf_threshold + 0.10,
        )

        # ── Step 4: Fuse Node C (3-class) ────────────────────────────────
        c_classes = ["belly_pain", "burping", "discomfort"]
        default_c = {c: 1.0 / 3 for c in c_classes}

        inputs_c = [
            ModalityInput("audio", w_audio_c, p_classes=audio_nodeC_proba or default_c),
            ModalityInput("bio", w_bio, p_classes=bio_node_c or default_c),
            ModalityInput("video", w_video, p_classes=video_node_c or default_c),
        ]
        post_c, conflict_c, used_c = _rw_poe_multiclass(inputs_c, c_classes)

        top_c = max(post_c, key=post_c.get)
        conf_c = post_c[top_c]

        node_c = NodeFusionResult(
            node="node_c",
            posterior=post_c,
            prediction=top_c,
            confidence=conf_c,
            conflict_score=conflict_c,
            modalities_used=used_c,
            is_uncertain=conf_c < self.conf_threshold,
        )

        # ── Step 5: Combine nodes into final cause probabilities ─────────
        # The final cause is determined by traversing the hierarchy:
        #   If hungry → "hungry"
        #   Else if tired → "tired"
        #   Else → Node C result (belly_pain / burping / discomfort)
        #
        # But we do this probabilistically, not as hard decisions:
        #   P(hungry)     = P(hungry from A)
        #   P(tired)      = P(not hungry) × P(tired from B)
        #   P(belly_pain) = P(not hungry) × P(not tired) × P(belly from C)
        #   P(burping)    = P(not hungry) × P(not tired) × P(burp from C)
        #   P(discomfort) = P(not hungry) × P(not tired) × P(disc from C)

        p_not_hungry = 1.0 - p_hungry
        p_not_tired = 1.0 - p_tired

        all_proba = {
            "hungry": p_hungry,
            "tired": p_not_hungry * p_tired,
            "belly_pain": p_not_hungry * p_not_tired * post_c.get("belly_pain", 0.33),
            "burping": p_not_hungry * p_not_tired * post_c.get("burping", 0.33),
            "discomfort": p_not_hungry * p_not_tired * post_c.get("discomfort", 0.34),
        }

        # Normalize to sum to 1
        total = sum(all_proba.values()) + EPS
        all_proba = {k: v / total for k, v in all_proba.items()}

        # Top cause
        top_cause = max(all_proba, key=all_proba.get)
        confidence = all_proba[top_cause]

        # ── Step 6: Detect conflicts and uncertainty ─────────────────────
        conflicts = []
        if conflict_a > self.conflict_threshold:
            conflicts.append(f"Node A: modalities disagree on hunger (conflict={conflict_a:.2f})")
        if conflict_b > self.conflict_threshold:
            conflicts.append(f"Node B: modalities disagree on tiredness (conflict={conflict_b:.2f})")
        if conflict_c > self.conflict_threshold:
            conflicts.append(f"Node C: modalities disagree on discomfort type (conflict={conflict_c:.2f})")

        is_uncertain = confidence < self.conf_threshold or len(conflicts) > 0

        # Apply cry gate discount: if cry detection is uncertain, reduce confidence
        if cry_gate_p < 0.90:
            confidence *= cry_gate_p
            if cry_gate_p < 0.70:
                is_uncertain = True

        # No modalities at all → always uncertain
        if not available:
            is_uncertain = True

        # ── Step 7: Build final result ────────────────────────────────────
        diagnosis = FinalDiagnosis(
            top_cause=top_cause,
            all_proba=all_proba,
            confidence=float(confidence),
            is_uncertain=is_uncertain,
            conflicts=conflicts,
            warnings=clinical_warnings or [],
            node_results={"node_a": node_a, "node_b": node_b, "node_c": node_c},
            modalities_available=available,
        )

        if self.verbose:
            print(f"\n  ── Fusion Result ──")
            print(f"  {diagnosis.summary()}")

        return diagnosis


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE: fuse from model outputs directly
# ═══════════════════════════════════════════════════════════════════════════════

def fuse_all(
    audio_a=None, audio_b=None, audio_c=None,
    bio=None, video=None, cry_gate=None,
    verbose=True,
) -> FinalDiagnosis:
    """Convenience function: takes model result objects directly.

    This is what the app pipeline calls. It extracts the right fields
    from each model's result object and passes them to FusionEngine.
    """
    engine = FusionEngine(verbose=verbose)

    # Extract audio predictions
    a_hungry = getattr(audio_a, 'p_hungry', None) if audio_a else None
    a_conf_a = getattr(audio_a, 'confidence', 0.5) if audio_a else 0.5
    a_tired = getattr(audio_b, 'p_tired', None) if audio_b else None
    a_conf_b = getattr(audio_b, 'confidence', 0.5) if audio_b else 0.5
    a_c_proba = getattr(audio_c, 'proba', None) if audio_c else None
    a_conf_c = getattr(audio_c, 'confidence', 0.33) if audio_c else 0.33

    # Extract bio predictions
    b_hungry = getattr(bio, 'node_a_hungry', None) if bio else None
    b_tired = getattr(bio, 'node_b_tired', None) if bio else None
    b_c = getattr(bio, 'node_c', None) if bio else None
    b_comp = getattr(bio, 'completeness', 0.0) if bio else 0.0
    b_warnings = getattr(bio, 'warnings', []) if bio else []

    # Extract video predictions
    v_hungry = getattr(video, 'node_a_hungry', None) if video else None
    v_tired = getattr(video, 'node_b_tired', None) if video else None
    v_c = getattr(video, 'node_c', None) if video else None
    v_rel = getattr(video, 'reliability', 0.0) if video else 0.0

    # Extract cry gate
    cg_p = getattr(cry_gate, 'p_cry', 1.0) if cry_gate else 1.0
    cg_q = getattr(cry_gate, 'quality_score', 1.0) if cry_gate else 1.0

    return engine.fuse(
        audio_nodeA_p_hungry=a_hungry,
        audio_nodeA_confidence=a_conf_a,
        audio_nodeB_p_tired=a_tired,
        audio_nodeB_confidence=a_conf_b,
        audio_nodeC_proba=a_c_proba,
        audio_nodeC_confidence=a_conf_c,
        bio_p_hungry=b_hungry,
        bio_p_tired=b_tired,
        bio_node_c=b_c,
        bio_completeness=b_comp,
        video_p_hungry=v_hungry,
        video_p_tired=v_tired,
        video_node_c=v_c,
        video_reliability=v_rel,
        cry_gate_p=cg_p,
        cry_gate_quality=cg_q,
        clinical_warnings=b_warnings,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Testing fusion.py")
    print("=" * 60)

    engine = FusionEngine(verbose=True)

    # ── Test 1: Audio + Bio → hungry (bio is strong) ──────────────────────
    print("\n  Test 1: Audio weak (0.56) + Bio strong (0.85) → hungry")
    d = engine.fuse(
        audio_nodeA_p_hungry=0.56,  audio_nodeA_confidence=0.55,
        audio_nodeB_p_tired=0.30,   audio_nodeB_confidence=0.60,
        audio_nodeC_proba={"belly_pain": 0.33, "burping": 0.33, "discomfort": 0.34},
        bio_p_hungry=0.85, bio_p_tired=0.20,
        bio_node_c={"belly_pain": 0.30, "burping": 0.20, "discomfort": 0.50},
        bio_completeness=0.85,
    )
    assert d.top_cause == "hungry", f"Expected hungry, got {d.top_cause}"
    print(f"\n  ✅ Top cause: {d.top_cause} (confidence={d.confidence:.3f})")
    print(f"     Bio rescued audio's weak signal!")

    # ── Test 2: Audio only (no bio, no video) ─────────────────────────────
    print("\n  " + "─" * 56)
    print("\n  Test 2: Audio only — no bio, no video")
    d = engine.fuse(
        audio_nodeA_p_hungry=0.65, audio_nodeA_confidence=0.60,
        audio_nodeB_p_tired=0.80, audio_nodeB_confidence=0.75,
        audio_nodeC_proba={"belly_pain": 0.20, "burping": 0.50, "discomfort": 0.30},
    )
    assert "audio" in d.modalities_available
    assert "bio" not in d.modalities_available
    print(f"\n  ✅ Works with audio only: {d.top_cause} (conf={d.confidence:.3f})")
    print(f"     Modalities: {d.modalities_available}")

    # ── Test 3: Bio only (no audio, no video) ─────────────────────────────
    print("\n  " + "─" * 56)
    print("\n  Test 3: Bio only — parent just entered context")
    d = engine.fuse(
        bio_p_hungry=0.90, bio_p_tired=0.10,
        bio_node_c={"belly_pain": 0.20, "burping": 0.10, "discomfort": 0.70},
        bio_completeness=0.80,
    )
    assert d.top_cause == "hungry"
    print(f"\n  ✅ Bio-only works: {d.top_cause} (conf={d.confidence:.3f})")

    # ── Test 4: All 3 modalities ──────────────────────────────────────────
    print("\n  " + "─" * 56)
    print("\n  Test 4: Audio + Bio + Video (all agree on tired)")
    d = engine.fuse(
        audio_nodeA_p_hungry=0.20, audio_nodeA_confidence=0.60,
        audio_nodeB_p_tired=0.85, audio_nodeB_confidence=0.80,
        audio_nodeC_proba={"belly_pain": 0.20, "burping": 0.30, "discomfort": 0.50},
        bio_p_hungry=0.15, bio_p_tired=0.80,
        bio_node_c={"belly_pain": 0.25, "burping": 0.25, "discomfort": 0.50},
        bio_completeness=0.70,
        video_p_hungry=0.10, video_p_tired=0.75,
        video_node_c={"belly_pain": 0.30, "burping": 0.30, "discomfort": 0.40},
        video_reliability=0.30,
    )
    assert d.top_cause == "tired"
    assert len(d.modalities_available) == 3
    print(f"\n  ✅ 3-modality fusion: {d.top_cause} (conf={d.confidence:.3f})")

    # ── Test 5: Conflict detection ────────────────────────────────────────
    print("\n  " + "─" * 56)
    print("\n  Test 5: Audio says hungry, Bio says NOT hungry → conflict")
    d = engine.fuse(
        audio_nodeA_p_hungry=0.85, audio_nodeA_confidence=0.70,
        audio_nodeB_p_tired=0.30, audio_nodeB_confidence=0.60,
        bio_p_hungry=0.15, bio_p_tired=0.30,
        bio_node_c={"belly_pain": 0.40, "burping": 0.30, "discomfort": 0.30},
        bio_completeness=0.80,
    )
    assert len(d.conflicts) > 0, "Expected conflict"
    print(f"\n  ✅ Conflict detected: {d.conflicts}")
    print(f"     Uncertain: {d.is_uncertain}")

    # ── Test 6: Nothing available ─────────────────────────────────────────
    print("\n  " + "─" * 56)
    print("\n  Test 6: No data at all → uniform uncertainty")
    d = engine.fuse()
    assert d.is_uncertain
    print(f"\n  ✅ No data: {d.top_cause} (conf={d.confidence:.3f}, uncertain={d.is_uncertain})")

    print(f"\n{'=' * 60}")
    print(f"  fusion.py — ALL TESTS PASSED ✅")
    print(f"{'=' * 60}")
