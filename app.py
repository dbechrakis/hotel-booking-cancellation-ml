"""Backward-compatible Streamlit application entry point."""

import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parent / "app" / "streamlit_app.py"),
    run_name="__main__",
)
