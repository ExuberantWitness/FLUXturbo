"""paths — parameterized sibling-repo locations (no hardcodes; env-overridable).

Resolves cc_wm_demo / cc_wm_research / FLUXturbo / le-wm from env vars or the default
E:/DATA/vscode siblings, and inserts them on sys.path (cc_wm_demo FIRST so `config`/`wm`
resolve to demo). Call `setup()` once at import.
"""
from __future__ import annotations
import os, sys
from pathlib import Path

VSCODE = Path(os.environ.get("DELTA_VSCODE", "E:/DATA/vscode"))
DEMO_DIR = Path(os.environ.get("DELTA_DEMO_DIR", VSCODE / "cc_wm_demo"))
RESEARCH_DIR = Path(os.environ.get("DELTA_RESEARCH_DIR", VSCODE / "cc_wm_research"))
FLUX_DIR = Path(os.environ.get("DELTA_FLUX_DIR", VSCODE / "FLUXturbo"))
LEWM_DIR = Path(os.environ.get("DELTA_LEWM_DIR", VSCODE / "le-wm"))

BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "output"; OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = "cuda"


def setup():
    """Insert sibling repos on sys.path so cc_wm_demo ends at position 0
    (its `config`/`wm`/`extract` win the name resolution)."""
    here = str(BASE)
    # insert in this order → after all insert(0,..), DEMO is at the front
    for p in [here, str(RESEARCH_DIR), str(FLUX_DIR), str(DEMO_DIR)]:
        if Path(p).exists() and p not in sys.path:
            sys.path.insert(0, p)


setup()
