"""Default configuration values and constants."""

from pathlib import Path

# Default directories
CONFIG_DIR = Path.home() / ".motorways"
CALIBRATION_FILE = CONFIG_DIR / "calibration.json"
LOG_DIR = CONFIG_DIR / "logs"
MODELS_DIR = CONFIG_DIR / "models"

# Default window settings
DEFAULT_TITLE_SUBSTR = "Mini Motorways"
DEFAULT_TARGET_FPS = 8
DEFAULT_INPUT_SIZE = (128, 128)

# Default grid settings
DEFAULT_GRID_SIZE = (32, 32)  # (height, width)

# Default toolbar button names
DEFAULT_TOOLBAR_BUTTONS = [
    "road",
    "bridge",
    "roundabout",
    "traffic_light",
    "motorway",
    "house"
]

# Performance settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_LOG_LEVEL = "INFO"

# Action timing settings
DEFAULT_CLICK_DURATION = 0.01
DEFAULT_DRAG_DURATION = 0.1
DEFAULT_MOVE_DURATION = 0.01

# Calibration validation thresholds
MIN_GRID_SIZE = 8
MAX_GRID_SIZE = 64
MIN_RATIO_DIFFERENCE = 0.1  # Minimum difference between x0r/x1r and y0r/y1r

# Model inference settings
DEFAULT_DEVICE = "cpu"  # Use "mps" for Apple Silicon if available
DEFAULT_DETERMINISTIC = True

# Logging settings
MAX_LOG_FILE_SIZE_MB = 100
MAX_LOG_FILES = 10

# Image processing settings
DEFAULT_INTERPOLATION = "bilinear"  # For cv2.resize
DEFAULT_NORMALIZATION = True  # Normalize pixel values to [0, 1]

# Error handling
DEFAULT_ERROR_RECOVERY_ATTEMPTS = 3
DEFAULT_PERMISSION_CHECK_INTERVAL = 60  # seconds

def ensure_directories() -> None:
    """Create default directories if they don't exist."""
    CONFIG_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)


def get_default_calibration_path() -> Path:
    """Get default calibration file path, creating directory if needed."""
    ensure_directories()
    return CALIBRATION_FILE


def get_default_log_path() -> Path:
    """Get default log directory path, creating it if needed."""
    ensure_directories()
    return LOG_DIR


def get_default_models_path() -> Path:
    """Get default models directory path, creating it if needed."""
    ensure_directories()
    return MODELS_DIR
