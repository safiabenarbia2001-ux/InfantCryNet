"""
predict_one.py — Predict the cause of one cry audio file.

THE MAIN ENTRY POINT for the application.

Usage examples:

    # Audio only
    python predict_one.py --audio cry.wav

    # Audio + bio context
    python predict_one.py --audio cry.wav --feed 180 --awake 40 --age 8

    # Bio context only (no audio)
    python predict_one.py --feed 180 --awake 40 --age 8 --temp 37.0

    # Full context
    python predict_one.py --audio cry.wav --feed 180 --awake 40 --age 8 --temp 37.0 --diaper 90 --feeding formula
"""

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.app.pipeline import InfantCryPipeline


def main():
    parser = argparse.ArgumentParser(
        description="InfantCryNet — Predict why the baby is crying",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_one.py --audio cry.wav
  python predict_one.py --audio cry.wav --feed 180 --age 8
  python predict_one.py --feed 180 --awake 40 --age 8 --temp 37.0
        """,
    )

    # Audio input
    parser.add_argument("--audio", default=None, help="Path to audio file (.wav/.ogg/.mp3)")

    # Bio context (parent enters in the app)
    parser.add_argument("--feed", type=float, default=None,
                        help="Minutes since last feed")
    parser.add_argument("--awake", type=float, default=None,
                        help="Minutes baby has been awake")
    parser.add_argument("--sleep", type=float, default=None,
                        help="Minutes since last sleep")
    parser.add_argument("--diaper", type=float, default=None,
                        help="Minutes since last diaper change")
    parser.add_argument("--temp", type=float, default=None,
                        help="Baby temperature in Celsius")
    parser.add_argument("--age", type=float, default=None,
                        help="Baby age in weeks")
    parser.add_argument("--feeding", default=None, choices=["breast", "formula"],
                        help="Feeding type")

    # Model directory
    parser.add_argument("--model-dir", default="models",
                        help="Directory with trained models")

    args = parser.parse_args()

    # Check we have at least something
    has_audio = args.audio is not None
    has_bio = any(v is not None for v in [
        args.feed, args.awake, args.sleep, args.diaper,
        args.temp, args.age, args.feeding,
    ])

    if not has_audio and not has_bio:
        print("\n  ❌ Please provide --audio and/or bio context (--feed, --awake, etc.)")
        print("  Run with --help for usage examples.")
        return

    # Check audio file exists
    if has_audio and not Path(args.audio).exists():
        print(f"\n  ❌ Audio file not found: {args.audio}")
        return

    # Build bio dict
    bio = None
    if has_bio:
        bio = {}
        if args.feed is not None:
            bio["minutes_since_feed"] = args.feed
        if args.awake is not None:
            bio["minutes_awake"] = args.awake
        if args.sleep is not None:
            bio["minutes_since_sleep"] = args.sleep
        if args.diaper is not None:
            bio["minutes_since_diaper"] = args.diaper
        if args.temp is not None:
            bio["temp_c"] = args.temp
        if args.age is not None:
            bio["age_weeks"] = args.age
        if args.feeding is not None:
            bio["feeding_type"] = args.feeding

    # Load pipeline
    print("\n  Loading models...")
    pipe = InfantCryPipeline(verbose=False)
    pipe.load_models(args.model_dir)

    # Predict
    result = pipe.predict(
        audio_path=args.audio,
        bio=bio,
    )

    # Print report
    result.print_report()


if __name__ == "__main__":
    main()
