#!/usr/bin/env python3
"""Entry point for DeTractor web interface."""

import uvicorn


def main():
    """Run the DeTractor web server."""
    print("Starting DeTractor Game Visualizer...")
    print("Open http://localhost:8000 in your browser")
    print()

    uvicorn.run(
        "web.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
