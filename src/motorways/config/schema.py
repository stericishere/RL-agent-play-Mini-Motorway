"""Pydantic models for configuration and data structures."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class Calibration(BaseModel):
    """Window calibration data storing grid boundaries as ratios."""

    grid_h: int = Field(..., gt=0, description="Grid height in cells")
    grid_w: int = Field(..., gt=0, description="Grid width in cells")
    x0r: float = Field(..., ge=0, le=1, description="Left boundary ratio (0-1)")
    y0r: float = Field(..., ge=0, le=1, description="Top boundary ratio (0-1)")
    x1r: float = Field(..., ge=0, le=1, description="Right boundary ratio (0-1)")
    y1r: float = Field(..., ge=0, le=1, description="Bottom boundary ratio (0-1)")
    toolbar: dict[str, tuple[float, float]] = Field(
        default_factory=dict,
        description="Toolbar button coordinates as ratios {tool_name: (x_ratio, y_ratio)}"
    )
    toolbar_region: Optional[dict[str, float]] = Field(
        default=None,
        description="Toolbar region boundaries as ratios {x0r, y0r, x1r, y1r} for dynamic button detection"
    )
    upgrades: dict[str, tuple[float, float]] = Field(
        default_factory=dict,
        description="Upgrade selection coordinates as ratios {position: (x_ratio, y_ratio)} - e.g., 'left', 'middle', 'right'"
    )
    upgrade_region: Optional[dict[str, float]] = Field(
        default=None,
        description="Upgrade selection region boundaries as ratios {x0r, y0r, x1r, y1r}"
    )

    @validator('x1r')
    def x1_greater_than_x0(cls, v, values):
        if 'x0r' in values and v <= values['x0r']:
            raise ValueError('x1r must be greater than x0r')
        return v

    @validator('y1r')
    def y1_greater_than_y0(cls, v, values):
        if 'y0r' in values and v <= values['y0r']:
            raise ValueError('y1r must be greater than y0r')
        return v

    def save(self, path: Path) -> None:
        """Save calibration to JSON file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(self.dict(), f, indent=2)
            logger.info(f"Saved calibration to {path}")
        except Exception as e:
            logger.error(f"Failed to save calibration to {path}: {e}")
            raise

    @classmethod
    def load(cls, path: Path) -> 'Calibration':
        """Load calibration from JSON file."""
        try:
            with open(path) as f:
                data = json.load(f)
            calibration = cls(**data)
            logger.info(f"Loaded calibration from {path}")
            return calibration
        except FileNotFoundError:
            logger.error(f"Calibration file not found: {path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load calibration from {path}: {e}")
            raise


class Settings(BaseModel):
    """Application settings and configuration."""

    title_substr: str = Field(default="Mini Motorways", description="Window title substring to search for")
    target_fps: int = Field(default=8, gt=0, le=60, description="Target FPS for agent loop")
    input_size: tuple[int, int] = Field(default=(128, 128), description="Model input image size (width, height)")
    model_path: Path = Field(..., description="Path to trained model file")
    dry_run: bool = Field(default=False, description="Run in dry-run mode (no actual clicks)")

    # Logging settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_dir: Path = Field(default=Path.home() / ".motorways" / "logs", description="Log directory")

    # Performance settings
    max_retries: int = Field(default=3, ge=0, description="Maximum retries for failed operations")
    timeout_seconds: float = Field(default=30.0, gt=0, description="Operation timeout in seconds")

    model_config = {"validate_assignment": True, "protected_namespaces": ()}

    def get_log_path(self) -> Path:
        """Get log file path for current date."""
        import datetime
        today = datetime.date.today().strftime("%Y%m%d")
        return self.log_dir / f"{today}.jsonl"


@dataclass
class Action:
    """Represents an agent action to be executed."""

    type: Literal["click", "drag", "noop", "toolbar", "upgrade"]
    r: Optional[int] = None  # Grid row
    c: Optional[int] = None  # Grid column
    path: Optional[list[tuple[int, int]]] = None  # Path for drag actions
    tool: Optional[str] = None  # Tool name for toolbar actions
    upgrade_choice: Optional[str] = None  # Upgrade choice: "left", "middle", "right"

    def __post_init__(self):
        """Validate action parameters."""
        if self.type == "click":
            if self.r is None or self.c is None:
                raise ValueError("Click action requires r and c coordinates")

        elif self.type == "drag":
            if self.path is None or len(self.path) < 2:
                raise ValueError("Drag action requires path with at least 2 points")

        elif self.type == "toolbar":
            if self.tool is None:
                raise ValueError("Toolbar action requires tool name")

        elif self.type == "upgrade":
            if self.upgrade_choice is None:
                raise ValueError("Upgrade action requires upgrade_choice")
            if self.upgrade_choice not in ["left", "middle", "right"]:
                raise ValueError("upgrade_choice must be 'left', 'middle', or 'right'")

        elif self.type == "noop":
            # No-op actions don't require additional parameters
            pass

        else:
            raise ValueError(f"Unknown action type: {self.type}")

    def to_dict(self) -> dict:
        """Convert action to dictionary for logging."""
        return {
            "type": self.type,
            "r": self.r,
            "c": self.c,
            "path": self.path,
            "tool": self.tool,
            "upgrade_choice": self.upgrade_choice
        }


class LogEntry(BaseModel):
    """Structure for JSONL log entries."""

    timestamp: float = Field(..., description="Unix timestamp")
    frame_sha1: Optional[str] = Field(None, description="SHA1 hash of captured frame")
    action: dict = Field(..., description="Action executed")
    mouse_xy: Optional[tuple[int, int]] = Field(None, description="Mouse coordinates used")
    fps: Optional[float] = Field(None, description="Current FPS")
    notes: Optional[str] = Field(None, description="Additional notes or error messages")

    def to_jsonl(self) -> str:
        """Convert to JSONL string."""
        return json.dumps(self.dict(), separators=(',', ':'))


class WindowInfo(BaseModel):
    """Information about a captured window."""

    window_id: int = Field(..., description="Window ID")
    title: str = Field(..., description="Window title")
    bounds: dict = Field(..., description="Window bounds (X, Y, Width, Height)")

    class Config:
        validate_assignment = True
