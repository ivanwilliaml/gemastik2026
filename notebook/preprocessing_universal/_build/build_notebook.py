"""Rebuild the current step-by-step notebook."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("build_examples.py")), run_name="__main__")
