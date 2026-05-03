"""
run.py — Start the BabyCare AI backend server.

Usage:
    cd C:\\Users\\hp\\Desktop\\babycare\\backend
    python run.py

    Or with options:
    python run.py --port 8000 --reload
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="BabyCare AI Backend Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    print(f"\n  🍼 BabyCare AI — Starting backend server")
    print(f"  → http://localhost:{args.port}")
    print(f"  → API docs: http://localhost:{args.port}/docs")
    print(f"  → Health: http://localhost:{args.port}/health\n")

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
