"""Project configuration: paths, device selection, and possession constants."""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
THIRD_PARTY_DIR = ROOT_DIR / "third_party"
SPORTS_REPO_DIR = THIRD_PARTY_DIR / "sports"
SOCCER_EXAMPLE_DIR = SPORTS_REPO_DIR / "examples" / "soccer"

# Roboflow's setup.sh downloads weights and sample clips here.
SOCCER_DATA_DIR = SOCCER_EXAMPLE_DIR / "data"
PLAYER_DETECTION_MODEL_PATH = SOCCER_DATA_DIR / "football-player-detection.pt"
BALL_DETECTION_MODEL_PATH = SOCCER_DATA_DIR / "football-ball-detection.pt"
PITCH_DETECTION_MODEL_PATH = SOCCER_DATA_DIR / "football-pitch-detection.pt"
DEFAULT_SOURCE_VIDEO = SOCCER_DATA_DIR / "2e57b9_0.mp4"

OUTPUTS_DIR = ROOT_DIR / "outputs"


# --- Device ------------------------------------------------------------------
def detect_device() -> str:
    """Return the best available torch device: 'cuda', then 'mps', else 'cpu'."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = detect_device()


# --- Possession --------------------------------------------------------------
# Owner-tuned. Do not change these here without the owner; the tuning harness
# passes overrides on the command line instead.
POSSESSION_MAX_DISTANCE_PX = 80
POSSESSION_SMOOTHING_WINDOW = 5
