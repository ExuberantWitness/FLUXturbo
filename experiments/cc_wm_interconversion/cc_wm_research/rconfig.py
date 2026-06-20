"""rconfig — cc_wm_research paths. (Named `rconfig` to avoid collision with cc_wm_demo/config.py
which both import as `config`.) Reuses cc_wm_demo's LeWM paths."""
import sys
from pathlib import Path

DEMO_DIR = Path("E:/DATA/vscode/cc_wm_demo")
# ensure cc_wm_demo/config.py is the module resolved by `import config`
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "output"; OUTPUT_DIR.mkdir(exist_ok=True)
PAPER_DIR = BASE / "paper"
DEVICE = "cuda"
SEED = 0

import config as demo_cfg  # noqa: E402  → cc_wm_demo/config.py (DEMO_DIR is on path)
REACHER_H5 = demo_cfg.DATA_DIR / "reacher_local.h5"
CKPT_DIR = demo_cfg.CKPT_DIR
EMBED_DIM = demo_cfg.EMBED_DIM
