"""
pipeline.py — Full inference pipeline with decision logic.

This is the BRAIN of the app. It:
    1. Loads all trained models once at startup
    2. Takes: audio file + optional bio context + optional video
    3. Runs the hierarchical pipeline: Stage 0 → Nodes A/B/C → Fusion
    4. Applies decision logic: what should the parent do?
    5. Returns a structured diagnosis with recommendations

Usage:
    pipe = InfantCryPipeline()           # loads all models
    pipe.load_models("models/")          # from your models folder

    result = pipe.predict(
        audio_path="path/to/cry.wav",
        bio={"minutes_since_feed": 180, "age_weeks": 8},
    )
    print(result.recommendation)
"""

import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.audio.audio_features import load_audio, extract_cause_features
from src.audio.cry_gate import CryGate, GateResult, GateZone
from src.audio.nodeA_model import NodeAModel, NodeAResult
from src.audio.nodeB_model import NodeBModel, NodeBResult
from src.audio.nodeC_model import NodeCModel, NodeCResult
from src.bio.bio_model import BioModel, BioInput, BioResult
from src.video.video_model import VideoModel, VideoResult
from src.fusion.fusion import FusionEngine, FinalDiagnosis, fuse_all


# ═══════════════════════════════════════════════════════════════════════════════
#  RECOMMENDATIONS — what to tell the parent
# ═══════════════════════════════════════════════════════════════════════════════

RECOMMENDATIONS = {
    "hungry": {
        "title": "Baby is likely hungry",
        "advice": [
            "Try offering a feed (breast or bottle)",
            "Look for hunger cues: sucking motions, rooting",
            "If baby refuses feed, consider other causes",
        ],
    },
    "tired": {
        "title": "Baby is likely tired",
        "advice": [
            "Create a calm, dark environment",
            "Try gentle rocking or swaddling",
            "Check how long baby has been awake",
            "Avoid overstimulation",
        ],
    },
    "belly_pain": {
        "title": "Baby may have belly pain (colic/gas)",
        "advice": [
            "Try gentle tummy massage in clockwise circles",
            "Bicycle the baby's legs gently",
            "Hold baby upright against your chest",
            "If symptoms persist, consult a pediatrician",
        ],
    },
    "burping": {
        "title": "Baby likely needs burping",
        "advice": [
            "Hold baby upright and pat their back gently",
            "Try different burping positions",
            "This is common after feeding",
        ],
    },
    "discomfort": {
        "title": "Baby seems uncomfortable",
        "advice": [
            "Check the diaper — may need changing",
            "Check room temperature (not too hot/cold)",
            "Check for tight clothing or tags irritating skin",
            "Look for any signs of illness",
        ],
    },
    "uncertain": {
        "title": "Multiple causes possible",
        "advice": [
            "Check the most common causes in order:",
            "1. Is baby hungry? (when was last feed?)",
            "2. Is baby tired? (how long awake?)",
            "3. Does baby need a diaper change?",
            "4. Is the temperature comfortable?",
            "If nothing helps, baby may just need comfort/holding",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE RESULT — everything the app needs
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    """Complete output of one prediction.

    This is what the app displays to the parent.
    """
    # Stage 0
    is_cry: bool
    p_cry: float
    cry_zone: str

    # Final diagnosis (None if not a cry)
    diagnosis: Optional[FinalDiagnosis]

    # Recommendation for the parent
    recommendation_title: str
    recommendation_advice: List[str]

    # Clinical warnings (from bio model)
    warnings: List[str]

    # Raw node results (for debugging/thesis)
    audio_nodeA: Optional[NodeAResult] = None
    audio_nodeB: Optional[NodeBResult] = None
    audio_nodeC: Optional[NodeCResult] = None
    bio_result: Optional[BioResult] = None
    video_result: Optional[VideoResult] = None

    # Timing
    elapsed_seconds: float = 0.0

    # What modalities were used
    modalities: List[str] = field(default_factory=list)

    def print_report(self):
        """Print a full report to console."""
        W = 56
        print(f"\n  {'═' * W}")
        print(f"  {'InfantCryNet — Prediction Report':^{W}}")
        print(f"  {'═' * W}")

        # Stage 0
        cry_icon = "✅ CRY" if self.is_cry else "❌ NOT A CRY"
        print(f"\n  Stage 0: P(cry) = {self.p_cry:.3f}  {cry_icon}")

        if not self.is_cry:
            print(f"\n  Audio does not contain crying.")
            print(f"  No further analysis needed.")
            return

        # Modalities
        print(f"  Modalities: {', '.join(self.modalities)}")

        # Node results (audio)
        if self.audio_nodeA:
            print(f"\n  Audio Node A: P(hungry) = {self.audio_nodeA.p_hungry:.3f}")
        if self.audio_nodeB:
            print(f"  Audio Node B: P(tired)  = {self.audio_nodeB.p_tired:.3f}")
        if self.audio_nodeC:
            print(f"  Audio Node C: {self.audio_nodeC.prediction} "
                  f"(conf={self.audio_nodeC.confidence:.3f})")

        # Bio result
        if self.bio_result:
            print(f"\n  Bio context:  P(hungry)={self.bio_result.node_a_hungry:.3f}  "
                  f"P(tired)={self.bio_result.node_b_tired:.3f}  "
                  f"completeness={self.bio_result.completeness:.0%}")

        # Fusion result
        if self.diagnosis:
            d = self.diagnosis
            print(f"\n  {'─' * W}")
            print(f"  FUSED RESULT: {d.top_cause.upper()} "
                  f"(confidence={d.confidence:.0%})")
            if d.is_uncertain:
                print(f"  ⚠️  Low confidence — multiple causes possible")

            print(f"\n  All probabilities:")
            for cause, p in sorted(d.all_proba.items(), key=lambda x: -x[1]):
                bar = "█" * int(p * 30)
                print(f"    {cause:<14} {p:6.1%}  {bar}")

        # Recommendation
        print(f"\n  {'─' * W}")
        print(f"  💡 {self.recommendation_title}")
        for advice in self.recommendation_advice:
            print(f"     • {advice}")

        # Warnings
        if self.warnings:
            print(f"\n  ⚠️  Clinical warnings:")
            for w in self.warnings:
                print(f"     {w}")

        print(f"\n  Time: {self.elapsed_seconds:.1f}s")
        print(f"  {'═' * W}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  THE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class InfantCryPipeline:
    """Full inference pipeline.

    Load once, predict many times.

    Handles any combination of modalities:
        Audio only         → microphone
        Audio + Bio        → microphone + parent input
        Audio + Bio + Video → microphone + input + camera
        Bio only           → just parent input (no audio)
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

        # Models (loaded by load_models)
        self.cry_gate: Optional[CryGate] = None
        self.node_a: Optional[NodeAModel] = None
        self.node_b: Optional[NodeBModel] = None
        self.node_c: Optional[NodeCModel] = None
        self.bio_model = BioModel(verbose=False)
        self.video_model = VideoModel(verbose=False)
        self.fusion = FusionEngine(verbose=False)

        self.is_loaded = False

    def load_models(self, model_dir: str = "models"):
        """Load all trained models from disk."""
        d = Path(model_dir)

        # Stage 0
        p = d / "stage0_cry_gate" / "cry_gate.pkl"
        if p.exists():
            self.cry_gate = CryGate.load(str(p))
            if self.verbose:
                print(f"  ✅ Loaded cry gate: {p}")
        else:
            if self.verbose:
                print(f"  ⚠️  Cry gate not found — will assume all audio is crying")

        # Node A
        p = d / "nodeA_hungry_vs_nonhungry" / "nodeA_model.pkl"
        if p.exists():
            self.node_a = NodeAModel.load(str(p))
            if self.verbose:
                print(f"  ✅ Loaded Node A: {p}")

        # Node B
        p = d / "nodeB_tired_vs_active" / "nodeB_model.pkl"
        if p.exists():
            self.node_b = NodeBModel.load(str(p))
            if self.verbose:
                print(f"  ✅ Loaded Node B: {p}")

        # Node C
        p = d / "nodeC_belly_burp_discomfort" / "nodeC_model.pkl"
        if p.exists():
            self.node_c = NodeCModel.load(str(p))
            if self.verbose:
                print(f"  ✅ Loaded Node C: {p}")

        self.is_loaded = True

    def predict(
        self,
        audio_path: Optional[str] = None,
        bio: Optional[Dict] = None,
        video_features=None,
    ) -> PipelineResult:
        """Run the full pipeline on one sample.

        Args:
            audio_path: path to audio file (None = no audio)
            bio: dict with any of these keys:
                 minutes_since_feed, minutes_awake, minutes_since_sleep,
                 minutes_since_diaper, temp_c, age_weeks, feeding_type
            video_features: VideoFeatures object (None = no video)
        """
        t0 = time.time()
        modalities = []
        warnings = []

        # ── STAGE 0: Cry gate ─────────────────────────────────────────────
        gate_result = None
        is_cry = True
        p_cry = 1.0
        cry_zone = "cry"

        if audio_path is not None:
            modalities.append("audio")
            if self.cry_gate is not None:
                gate_result = self.cry_gate.predict(audio_path)
                is_cry = gate_result.should_proceed
                p_cry = gate_result.p_cry
                cry_zone = gate_result.zone.value
            else:
                # No cry gate → assume it's a cry
                is_cry = True
                p_cry = 1.0
        elif bio is not None:
            # No audio but bio available → skip cry gate
            is_cry = True
            p_cry = 1.0
        else:
            # Nothing available
            return PipelineResult(
                is_cry=False, p_cry=0.0, cry_zone="non_cry",
                diagnosis=None,
                recommendation_title="No input provided",
                recommendation_advice=["Please provide an audio file or bio context"],
                warnings=[], elapsed_seconds=time.time() - t0,
            )

        # If not a cry → stop
        if not is_cry:
            return PipelineResult(
                is_cry=False, p_cry=p_cry, cry_zone=cry_zone,
                diagnosis=None,
                recommendation_title="Not a cry",
                recommendation_advice=["The audio does not appear to contain crying"],
                warnings=[], elapsed_seconds=time.time() - t0,
                modalities=modalities,
            )

        # ── AUDIO NODES ───────────────────────────────────────────────────
        result_a = None
        result_b = None
        result_c = None

        if audio_path is not None:
            if self.node_a is not None:
                result_a = self.node_a.predict(audio_path)
            if self.node_b is not None:
                result_b = self.node_b.predict(audio_path)
            if self.node_c is not None:
                result_c = self.node_c.predict(audio_path)

        # ── BIO MODEL ─────────────────────────────────────────────────────
        bio_result = None
        if bio is not None and len(bio) > 0:
            modalities.append("bio")
            bio_input = BioInput(
                minutes_since_feed=bio.get("minutes_since_feed"),
                minutes_since_sleep=bio.get("minutes_since_sleep"),
                minutes_awake=bio.get("minutes_awake"),
                minutes_since_diaper=bio.get("minutes_since_diaper"),
                temp_c=bio.get("temp_c"),
                age_weeks=bio.get("age_weeks"),
                feeding_type=bio.get("feeding_type"),
            )
            bio_result = self.bio_model.predict(bio_input)
            warnings.extend(bio_result.warnings)

        # ── VIDEO MODEL ───────────────────────────────────────────────────
        video_result = None
        if video_features is not None:
            modalities.append("video")
            video_result = self.video_model.predict_from_features(
                video_features, is_simulated=False)
        # If no video → fusion will handle it (reliability=0)

        # ── FUSION ────────────────────────────────────────────────────────
        diagnosis = fuse_all(
            audio_a=result_a,
            audio_b=result_b,
            audio_c=result_c,
            bio=bio_result,
            video=video_result,
            cry_gate=gate_result,
            verbose=False,
        )

        # ── DECISION LOGIC ────────────────────────────────────────────────
        if diagnosis.is_uncertain:
            rec = RECOMMENDATIONS["uncertain"]
        else:
            rec = RECOMMENDATIONS.get(diagnosis.top_cause, RECOMMENDATIONS["uncertain"])

        # If there are urgent clinical warnings, override recommendation
        urgent = [w for w in warnings if "urgent" in w.lower()]
        if urgent:
            rec = {
                "title": "⚠️ Medical attention may be needed",
                "advice": [w for w in urgent] + [
                    "Please consult a healthcare provider",
                ],
            }

        elapsed = time.time() - t0

        return PipelineResult(
            is_cry=True,
            p_cry=p_cry,
            cry_zone=cry_zone,
            diagnosis=diagnosis,
            recommendation_title=rec["title"],
            recommendation_advice=rec["advice"],
            warnings=warnings,
            audio_nodeA=result_a,
            audio_nodeB=result_b,
            audio_nodeC=result_c,
            bio_result=bio_result,
            video_result=video_result,
            elapsed_seconds=elapsed,
            modalities=modalities,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST (no audio files needed — tests with bio only)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Testing pipeline.py")
    print("=" * 60)

    pipe = InfantCryPipeline(verbose=True)

    # Try loading models (may not exist in test environment)
    print("\n  Loading models...")
    pipe.load_models("models")

    # ── Test 1: Bio only (no audio needed) ────────────────────────────────
    print("\n  Test 1: Bio only — baby fed 3 hours ago")
    result = pipe.predict(
        bio={"minutes_since_feed": 180, "age_weeks": 8, "feeding_type": "formula",
             "minutes_awake": 40, "temp_c": 37.0},
    )
    result.print_report()

    # ── Test 2: Bio only — tired baby ─────────────────────────────────────
    print("\n  Test 2: Bio only — baby awake for 2 hours")
    result = pipe.predict(
        bio={"minutes_since_feed": 30, "minutes_awake": 120, "age_weeks": 12},
    )
    result.print_report()

    # ── Test 3: Bio only — fever ──────────────────────────────────────────
    print("\n  Test 3: Bio only — baby has fever")
    result = pipe.predict(
        bio={"temp_c": 39.0, "age_weeks": 3, "minutes_since_feed": 60},
    )
    result.print_report()

    # ── Test 4: No input ──────────────────────────────────────────────────
    print("\n  Test 4: No input at all")
    result = pipe.predict()
    assert result.is_cry is False
    print(f"  ✅ Correctly rejected: {result.recommendation_title}")

    print(f"\n{'=' * 60}")
    print(f"  pipeline.py — TESTS PASSED ✅")
    print(f"{'=' * 60}")
