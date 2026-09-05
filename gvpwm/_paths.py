"""Location of the upstream DINO-WM checkout."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)

DINO_WM_ROOT = Path(os.environ.get("DINO_WM_ROOT", "dino_wm")).expanduser()
if not DINO_WM_ROOT.is_absolute():
    DINO_WM_ROOT = PROJECT_ROOT / DINO_WM_ROOT
DINO_WM_ROOT = DINO_WM_ROOT.resolve()

def configure_dino_wm_path():
    # Upstream DINO-WM uses top-level imports such as `env` and `datasets`.
    if str(DINO_WM_ROOT) not in sys.path:
        sys.path.insert(0, str(DINO_WM_ROOT))
