"""Central config: paths and constants for the CC⇌WM demo (le-wm reacher)."""
from pathlib import Path

# ── layout ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
OUTPUT_DIR = BASE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# le-wm / FLUXturbo siblings
LEWM_DIR = Path("E:/DATA/vscode/le-wm")
FLUXTURBO_DIR = Path("E:/DATA/vscode/FLUXturbo")

# HF checkpoint (downloaded by setup)
CKPT_DIR = DATA_DIR / "lewm-reacher"
WEIGHTS_PT = CKPT_DIR / "weights.pt"
CONFIG_JSON = CKPT_DIR / "config.json"

# dataset (filled in by setup / discovery)
REACHER_H5 = DATA_DIR / "reacher.h5"

# ── model constants (from le-wm config/train/model/lewm.yaml + lewm.yaml) ─
EMBED_DIM = 192
IMG_SIZE = 224
HISTORY_SIZE = 3
NUM_PREDS = 1

# ImageNet stats (matches le-wm utils.get_img_preprocessor)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEVICE = "cuda"  # set to "cpu" as fallback

# probing thresholds
R2_HIGH = 0.7   # → concept/numerical fact (verified)
R2_LOW = 0.3    # → representational_limitation bottleneck
