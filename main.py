"""
main.py — AI-Powered Multimodal Baby Cry Detection
═══════════════════════════════════════════════════

THE MAIN ENTRY POINT for the entire system.

This script demonstrates and runs ALL capabilities:
    1. Load trained models
    2. Test with real audio files
    3. Enter biological context interactively
    4. Process video features
    5. Run trimodal fusion (audio + bio + video)
    6. Show predictions with confidence and recommendations
    7. Run full evaluation
    8. Generate thesis figures

Modes:
    python main.py                   → Interactive menu
    python main.py --mode predict --audio cry.wav
    python main.py --mode predict --audio cry.wav --feed 180 --age 8
    python main.py --mode evaluate
    python main.py --mode demo

Authors: BENARBIA Safia, Pr. KHALFI M.F., Dr. BOUABSSA Wahiba
Djillali LIABES University, Sidi Bel Abbès, Algeria
"""

import argparse
import sys
import time
import os
from pathlib import Path
from typing import Dict, Optional

# ── Path fix (works from any working directory) ───────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from src.audio.audio_features import load_audio, extract_cause_features, check_quality
from src.audio.cry_gate import CryGate
from src.audio.nodeA_model import NodeAModel, NodeAResult
from src.audio.nodeB_model import NodeBModel, NodeBResult
from src.audio.nodeC_model import NodeCModel, NodeCResult
from src.bio.bio_model import BioModel, BioInput, BioResult
from src.bio.bio_priors import get_age_profile, get_cause_priors
from src.bio.clinical_rules import run_all_checks, Urgency
from src.video.video_model import VideoModel, VideoFeatures, VideoResult, VideoSimulator
from src.fusion.fusion import FusionEngine, FinalDiagnosis, fuse_all
from src.app.pipeline import InfantCryPipeline, PipelineResult, RECOMMENDATIONS

W = 64


def hdr(title, char="═"):
    print(f"\n  {char * W}")
    print(f"  {title:^{W}}")
    print(f"  {char * W}")


def sep(char="─"):
    print(f"  {char * W}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_models(model_dir: str = None) -> InfantCryPipeline:
    """Load the full pipeline with all trained models."""
    if model_dir is None:
        model_dir = str(_THIS_DIR / "models")

    hdr("Loading Models")
    pipe = InfantCryPipeline(verbose=True)
    pipe.load_models(model_dir)
    print(f"\n  Models directory: {model_dir}")

    # Report what's loaded
    loaded = []
    if pipe.cry_gate is not None:
        loaded.append("Stage 0 (Cry Gate)")
    if pipe.node_a is not None:
        loaded.append("Node A (Hungry)")
    if pipe.node_b is not None:
        loaded.append("Node B (Tired)")
    if pipe.node_c is not None:
        loaded.append("Node C (Discomfort)")
    loaded.append("Bio Model (rule-based, always available)")
    loaded.append("Video Model (feature-based, always available)")
    loaded.append("Fusion Engine (RW-PoE, always available)")

    print(f"\n  Loaded {len(loaded)} components:")
    for name in loaded:
        print(f"    ✅ {name}")

    return pipe


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE BIO INPUT
# ═══════════════════════════════════════════════════════════════════════════════

def ask_bio_interactive() -> Dict:
    """Ask the parent for biological context — what the app would collect."""
    hdr("Biological Context Input", "─")
    print("  Enter baby information (press Enter to skip any field):\n")

    def ask_float(prompt, default=None):
        try:
            val = input(f"  {prompt}: ").strip()
            if val == "":
                return default
            return float(val)
        except (ValueError, EOFError):
            return default

    def ask_str(prompt, choices=None, default=None):
        try:
            hint = f" ({'/'.join(choices)})" if choices else ""
            val = input(f"  {prompt}{hint}: ").strip()
            if val == "":
                return default
            if choices and val.lower() not in [c.lower() for c in choices]:
                print(f"    ⚠️  Invalid choice, using default: {default}")
                return default
            return val.lower()
        except EOFError:
            return default

    bio = {}

    # Feeding
    val = ask_float("Minutes since last feed (e.g., 180 = 3 hours)")
    if val is not None:
        bio["minutes_since_feed"] = val

    # Sleep
    val = ask_float("Minutes baby has been awake (e.g., 90)")
    if val is not None:
        bio["minutes_awake"] = val

    # Diaper
    val = ask_float("Minutes since diaper change (e.g., 120)")
    if val is not None:
        bio["minutes_since_diaper"] = val

    # Temperature
    val = ask_float("Baby temperature in °C (e.g., 37.0)")
    if val is not None:
        bio["temp_c"] = val

    # Age
    val = ask_float("Baby age in weeks (e.g., 8)")
    if val is not None:
        bio["age_weeks"] = val

    # Feeding type
    val = ask_str("Feeding type", choices=["breast", "formula"], default=None)
    if val is not None:
        bio["feeding_type"] = val

    if not bio:
        print("\n  ⚠️  No bio context provided — using audio only")
    else:
        print(f"\n  ✅ Bio context: {len(bio)} fields entered")
        for k, v in bio.items():
            print(f"    {k}: {v}")

    return bio


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE VIDEO INPUT
# ═══════════════════════════════════════════════════════════════════════════════

def ask_video_interactive() -> Optional[VideoFeatures]:
    """Ask for video behavioral features — what a CNN would extract."""
    hdr("Video Features Input", "─")
    print("  Enter observed behavioral features (0.0 to 1.0).")
    print("  These would be extracted by a CNN in the app.")
    print("  Press Enter to skip (= no video available).\n")

    def ask_feature(name, description):
        try:
            val = input(f"  {name} ({description}) [0-1]: ").strip()
            if val == "":
                return None
            return max(0.0, min(1.0, float(val)))
        except (ValueError, EOFError):
            return None

    features = {}
    fields = [
        ("mouth_movement", "sucking/rooting motions → hunger"),
        ("eye_rubbing", "rubbing eyes → tiredness"),
        ("eye_closing", "heavy eyelids, closing → tiredness"),
        ("legs_pulled_up", "legs to belly → pain/colic"),
        ("back_arching", "arching back → pain"),
        ("movement_intensity", "overall body movement"),
        ("facial_grimace", "pain expression on face"),
        ("yawning", "yawning detected → tiredness"),
    ]

    first = ask_feature(fields[0][0], fields[0][1])
    if first is None:
        print("\n  ⚠️  No video input — fusion will use audio + bio only")
        return None

    features[fields[0][0]] = first
    for name, desc in fields[1:]:
        val = ask_feature(name, desc)
        features[name] = val if val is not None else 0.0

    print(f"\n  ✅ Video features entered: {len(features)} values")
    return VideoFeatures(**features)


# ═══════════════════════════════════════════════════════════════════════════════
#  PREDICT WITH FULL REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def predict_with_report(
    pipe: InfantCryPipeline,
    audio_path: Optional[str] = None,
    bio: Optional[Dict] = None,
    video_features: Optional[VideoFeatures] = None,
):
    """Run prediction and show a comprehensive report."""
    hdr("Running Prediction Pipeline")

    # Show what modalities are available
    mods = []
    if audio_path:
        mods.append(f"Audio: {Path(audio_path).name}")
    if bio and len(bio) > 0:
        mods.append(f"Bio: {len(bio)} fields")
    if video_features:
        mods.append("Video: 8 features")

    if not mods:
        print("\n  ❌ No input provided!")
        return

    print(f"\n  Input modalities:")
    for m in mods:
        print(f"    → {m}")

    # Run prediction
    result = pipe.predict(
        audio_path=audio_path,
        bio=bio,
        video_features=video_features,
    )

    # Show full report
    result.print_report()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO MODE — shows all capabilities without user input
# ═══════════════════════════════════════════════════════════════════════════════

def run_demo(pipe: InfantCryPipeline):
    """Demonstrate all system capabilities with example scenarios."""
    hdr("DEMO: All System Capabilities")

    # ── Demo 1: Bio only ──────────────────────────────────────────────────
    print("\n")
    hdr("Demo 1: Bio Only — Baby fed 3 hours ago", "─")
    print("  (Parent opens app, enters context, no audio recorded yet)")
    result = pipe.predict(bio={
        "minutes_since_feed": 180,
        "age_weeks": 8,
        "feeding_type": "formula",
        "minutes_awake": 40,
        "temp_c": 37.0,
        "minutes_since_diaper": 60,
    })
    result.print_report()

    # ── Demo 2: Bio only — tired ──────────────────────────────────────────
    hdr("Demo 2: Bio Only — Baby awake for 2 hours", "─")
    result = pipe.predict(bio={
        "minutes_since_feed": 30,
        "minutes_awake": 120,
        "age_weeks": 12,
        "temp_c": 37.0,
    })
    result.print_report()

    # ── Demo 3: Fever emergency ───────────────────────────────────────────
    hdr("Demo 3: Clinical Emergency — Neonate with Fever", "─")
    print("  (Safety override should trigger regardless of classification)")
    result = pipe.predict(bio={
        "temp_c": 38.5,
        "age_weeks": 2,
        "minutes_since_feed": 90,
    })
    result.print_report()

    # ── Demo 4: Bio + Video (simulated) ───────────────────────────────────
    hdr("Demo 4: Bio + Video — Hungry baby with mouth movements", "─")
    video_feats = VideoFeatures(
        mouth_movement=0.85,
        eye_rubbing=0.10,
        eye_closing=0.15,
        legs_pulled_up=0.05,
        back_arching=0.05,
        movement_intensity=0.50,
        facial_grimace=0.20,
        yawning=0.05,
    )
    result = pipe.predict(
        bio={
            "minutes_since_feed": 200,
            "age_weeks": 10,
            "feeding_type": "breast",
            "minutes_awake": 50,
        },
        video_features=video_feats,
    )
    result.print_report()

    # ── Demo 5: Audio file (if any exist) ─────────────────────────────────
    # Try to find a real audio file for demo
    audio_dirs = [
        _THIS_DIR / "data" / "interim" / "nodeA" / "hungry",
        _THIS_DIR / "data" / "interim" / "stage_0" / "cry",
    ]
    demo_audio = None
    for d in audio_dirs:
        if d.exists():
            files = list(d.glob("*.wav")) + list(d.glob("*.ogg"))
            if files:
                demo_audio = str(files[0])
                break

    if demo_audio:
        hdr(f"Demo 5: Real Audio + Bio", "─")
        print(f"  Audio file: {Path(demo_audio).name}")
        result = pipe.predict(
            audio_path=demo_audio,
            bio={
                "minutes_since_feed": 180,
                "age_weeks": 8,
                "temp_c": 37.0,
                "minutes_awake": 45,
            },
        )
        result.print_report()
    else:
        hdr("Demo 5: Skipped (no audio files found)", "─")
        print(f"  Place .wav files in data/interim/nodeA/hungry/ to test")

    # ── Demo 6: Missing modality handling ─────────────────────────────────
    hdr("Demo 6: No Input — System Handles Gracefully", "─")
    result = pipe.predict()
    print(f"  Result: is_cry={result.is_cry}")
    print(f"  Message: {result.recommendation_title}")

    hdr("Demo Complete — All Modes Demonstrated")


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATION MODE
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation():
    """Run full evaluation and generate plots."""
    hdr("Running Full Evaluation")

    # Import evaluation modules
    from src.evaluation.eval_all import main as eval_main
    from src.evaluation.plots import main as plots_main

    print("\n  Step 1: Evaluating all nodes + fusion...")
    eval_main()

    print("\n  Step 2: Generating thesis figures...")
    plots_main()

    print("\n  ✅ Evaluation complete!")
    print(f"  Results: {_THIS_DIR / 'results'}")
    print(f"  Figures: {_THIS_DIR / 'results' / 'figures'}")


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE MENU
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_menu(pipe: InfantCryPipeline):
    """Interactive menu for exploring the system."""
    while True:
        hdr("AI-Powered Multimodal Baby Cry Detection")
        print(f"""
  ┌────────────────────────────────────────────────────────────┐
  │  1. Predict from AUDIO file                                │
  │  2. Predict from AUDIO + BIO context                       │
  │  3. Predict from AUDIO + BIO + VIDEO features              │
  │  4. Predict from BIO context only (no audio)               │
  │  5. Predict from BIO + VIDEO only                          │
  │  6. Run DEMO (all capabilities)                            │
  │  7. Run EVALUATION (all nodes + fusion + plots)            │
  │  8. System INFO (models, data, architecture)               │
  │  0. Exit                                                   │
  └────────────────────────────────────────────────────────────┘
        """)

        try:
            choice = input("  Choose [0-8]: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "0":
            print("\n  Goodbye! 👶")
            break

        elif choice == "1":
            # Audio only
            try:
                path = input("\n  Audio file path: ").strip()
                if path and Path(path).exists():
                    predict_with_report(pipe, audio_path=path)
                else:
                    print(f"  ❌ File not found: {path}")
            except (EOFError, KeyboardInterrupt):
                continue

        elif choice == "2":
            # Audio + Bio
            try:
                path = input("\n  Audio file path: ").strip()
                if not path or not Path(path).exists():
                    print(f"  ❌ File not found: {path}")
                    continue
                bio = ask_bio_interactive()
                predict_with_report(pipe, audio_path=path, bio=bio)
            except (EOFError, KeyboardInterrupt):
                continue

        elif choice == "3":
            # Audio + Bio + Video
            try:
                path = input("\n  Audio file path: ").strip()
                if not path or not Path(path).exists():
                    print(f"  ❌ File not found: {path}")
                    continue
                bio = ask_bio_interactive()
                video = ask_video_interactive()
                predict_with_report(pipe, audio_path=path, bio=bio,
                                    video_features=video)
            except (EOFError, KeyboardInterrupt):
                continue

        elif choice == "4":
            # Bio only
            bio = ask_bio_interactive()
            if bio:
                predict_with_report(pipe, bio=bio)

        elif choice == "5":
            # Bio + Video
            bio = ask_bio_interactive()
            video = ask_video_interactive()
            if bio:
                predict_with_report(pipe, bio=bio, video_features=video)

        elif choice == "6":
            run_demo(pipe)

        elif choice == "7":
            run_evaluation()

        elif choice == "8":
            show_system_info(pipe)

        # Pause before returning to menu
        if choice != "0":
            try:
                input("\n  Press Enter to continue...")
            except (EOFError, KeyboardInterrupt):
                pass


def show_system_info(pipe: InfantCryPipeline):
    """Display system architecture and loaded models."""
    hdr("System Information")

    print(f"""
  Project: AI-Powered Multimodal Baby Cry Detection
  Authors: BENARBIA Safia, KHALFI M.F., BOUABSSA Wahiba
  University: Djillali LIABES, Sidi Bel Abbès, Algeria
  Clinical Partner: CHU Sidi Bel Abbès

  ┌─ Architecture ─────────────────────────────────────────────┐
  │                                                            │
  │  Audio ──→ Stage 0 (cry/non-cry)                          │
  │            ├── Node A (hungry / non-hungry)                │
  │            ├── Node B (tired / active)                     │
  │            └── Node C (belly_pain / burping / discomfort)  │
  │                                                            │
  │  Bio ────→ Age-adaptive priors + Clinical rules            │
  │            P(hungry), P(tired), P(discomfort sub-types)    │
  │                                                            │
  │  Video ──→ 8 behavioral features                           │
  │            P(hungry), P(tired), P(discomfort sub-types)    │
  │                                                            │
  │  Fusion ─→ RW-PoE (Reliability-Weighted Product of Experts)│
  │            Hierarchical combination → Final diagnosis       │
  │                                                            │
  └────────────────────────────────────────────────────────────┘

  ┌─ Models ───────────────────────────────────────────────────┐
  │  Cry Gate:   {"✅ loaded" if pipe.cry_gate else "❌ not found":>10}   (AUC = 0.996)      │
  │  Node A:     {"✅ loaded" if pipe.node_a else "❌ not found":>10}   (AUC = 0.560)      │
  │  Node B:     {"✅ loaded" if pipe.node_b else "❌ not found":>10}   (AUC = 0.799)      │
  │  Node C:     {"✅ loaded" if pipe.node_c else "❌ not found":>10}   (Acc = 0.500)      │
  │  Bio Model:  ✅ always   (rule-based)           │
  │  Video:      ✅ always   (feature-based)         │
  │  Fusion:     ✅ always   (RW-PoE)               │
  └────────────────────────────────────────────────────────────┘

  ┌─ Key Results ──────────────────────────────────────────────┐
  │  Audio only:          56.8%  accuracy                      │
  │  Audio + Bio:         86.5%  accuracy  (+29.7%)            │
  │  Audio + Bio + Video: 91.6%  accuracy  (+34.8%)            │
  │  Non-hungry accuracy: 15.4% → 100.0%  (+84.6%)            │
  └────────────────────────────────────────────────────────────┘

  ┌─ Data Sources ─────────────────────────────────────────────┐
  │  Audio: Cry Sense dataset (1,105 samples)                  │
  │  Video: 2,390 recordings (6.6h), 165 annotated clips       │
  │  Bio:   Clinical data from CHU Sidi Bel Abbès              │
  └────────────────────────────────────────────────────────────┘
    """)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog="AI-Powered Multimodal Baby Cry Detection",
        description="Hierarchical Bayesian Multimodal Infant Cry Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              → Interactive menu
  python main.py --mode demo                  → Run all demos
  python main.py --mode predict --audio cry.wav
  python main.py --mode predict --audio cry.wav --feed 180 --age 8
  python main.py --mode predict --feed 180 --awake 120 --age 12
  python main.py --mode evaluate              → Full evaluation + plots
        """,
    )

    parser.add_argument("--mode", default="menu",
                        choices=["menu", "predict", "demo", "evaluate", "info"],
                        help="Running mode")
    parser.add_argument("--model-dir", default=None,
                        help="Models directory (default: <project>/models)")

    # Audio
    parser.add_argument("--audio", default=None, help="Path to audio file")

    # Bio context
    parser.add_argument("--feed", type=float, default=None,
                        help="Minutes since last feed")
    parser.add_argument("--awake", type=float, default=None,
                        help="Minutes baby has been awake")
    parser.add_argument("--diaper", type=float, default=None,
                        help="Minutes since diaper change")
    parser.add_argument("--temp", type=float, default=None,
                        help="Baby temperature °C")
    parser.add_argument("--age", type=float, default=None,
                        help="Baby age in weeks")
    parser.add_argument("--feeding", default=None,
                        choices=["breast", "formula"],
                        help="Feeding type")

    # Video (manual feature input for testing)
    parser.add_argument("--video-mouth", type=float, default=None,
                        help="Mouth movement intensity [0-1]")
    parser.add_argument("--video-eyes", type=float, default=None,
                        help="Eye rubbing intensity [0-1]")
    parser.add_argument("--video-legs", type=float, default=None,
                        help="Legs pulled up intensity [0-1]")

    args = parser.parse_args()

    # ── Banner ────────────────────────────────────────────────────────────
    hdr("AI-Powered Multimodal Baby Cry Detection")
    print(f"  Authors: BENARBIA Safia, KHALFI M.F., BOUABSSA Wahiba")
    print(f"  Djillali LIABES University, Sidi Bel Abbès, Algeria")
    print(f"  Mode: {args.mode}  |  Python {sys.version.split()[0]}")

    # ── Load models ───────────────────────────────────────────────────────
    pipe = load_all_models(args.model_dir)

    # ── Route to mode ─────────────────────────────────────────────────────
    if args.mode == "menu":
        interactive_menu(pipe)

    elif args.mode == "demo":
        run_demo(pipe)

    elif args.mode == "info":
        show_system_info(pipe)

    elif args.mode == "evaluate":
        run_evaluation()

    elif args.mode == "predict":
        # Build bio dict from CLI args
        bio = {}
        if args.feed is not None:
            bio["minutes_since_feed"] = args.feed
        if args.awake is not None:
            bio["minutes_awake"] = args.awake
        if args.diaper is not None:
            bio["minutes_since_diaper"] = args.diaper
        if args.temp is not None:
            bio["temp_c"] = args.temp
        if args.age is not None:
            bio["age_weeks"] = args.age
        if args.feeding is not None:
            bio["feeding_type"] = args.feeding

        # Build video features from CLI args
        video_feats = None
        if args.video_mouth is not None:
            video_feats = VideoFeatures(
                mouth_movement=args.video_mouth or 0.0,
                eye_rubbing=args.video_eyes or 0.0,
                eye_closing=0.0,
                legs_pulled_up=args.video_legs or 0.0,
                back_arching=0.0,
                movement_intensity=0.3,
                facial_grimace=0.2,
                yawning=0.0,
            )

        # Must have at least audio or bio
        if not args.audio and not bio:
            print("\n  ❌ Provide --audio and/or bio context (--feed, --age, etc.)")
            return

        if args.audio and not Path(args.audio).exists():
            print(f"\n  ❌ Audio file not found: {args.audio}")
            return

        predict_with_report(
            pipe,
            audio_path=args.audio,
            bio=bio if bio else None,
            video_features=video_feats,
        )


if __name__ == "__main__":
    main()
