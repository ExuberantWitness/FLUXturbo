"""Deprecated shim — use `rconfig` instead. (Kept name `config` would collide with
cc_wm_demo/config.py; this file only re-exports rconfig and is not imported as `config`
because cc_wm_demo is placed first on sys.path by callers.)"""
from rconfig import *  # noqa: F401,F403
from rconfig import REACHER_H5, CKPT_DIR, EMBED_DIM, OUTPUT_DIR, PAPER_DIR, DEVICE, SEED  # noqa: F401
