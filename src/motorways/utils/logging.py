"""JSONL logging utilities for agent runs."""

import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from motorways.config.schema import Action, LogEntry

logger = logging.getLogger(__name__)


class JSONLLogger:
    """JSONL logger for agent episode data."""

    def __init__(self, log_path: Path):
        """Initialize JSONL logger.

        Args:
            log_path: Path to JSONL log file
        """
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"JSONL logger initialized: {log_path}")

    def log_action(
        self,
        action: Action,
        frame: Optional[np.ndarray] = None,
        mouse_xy: Optional[tuple[int, int]] = None,
        fps: Optional[float] = None,
        notes: Optional[str] = None
    ) -> None:
        """Log an action execution.

        Args:
            action: Action that was executed
            frame: Optional captured frame for hash generation
            mouse_xy: Mouse coordinates used
            fps: Current FPS
            notes: Additional notes or error messages
        """
        try:
            # Generate frame hash if frame provided
            frame_sha1 = None
            if frame is not None:
                frame_bytes = frame.tobytes()
                frame_sha1 = hashlib.sha1(frame_bytes).hexdigest()

            # Create log entry
            log_entry = LogEntry(
                timestamp=time.time(),
                frame_sha1=frame_sha1,
                action=action.to_dict(),
                mouse_xy=mouse_xy,
                fps=fps,
                notes=notes
            )

            # Write to file
            with open(self.log_path, 'a') as f:
                f.write(log_entry.to_jsonl() + '\n')

            logger.debug(f"Logged action: {action.type}")

        except Exception as e:
            logger.error(f"Failed to log action: {e}")

    def log_event(
        self,
        event_type: str,
        details: Optional[dict] = None,
        notes: Optional[str] = None
    ) -> None:
        """Log a general event.

        Args:
            event_type: Type of event (e.g., "calibration", "error", "start", "stop")
            details: Additional event details
            notes: Event notes
        """
        try:
            from motorways.config.schema import Action

            # Create a special action for events
            Action(type="noop")
            event_details = {
                "event_type": event_type,
                "details": details or {}
            }

            log_entry = LogEntry(
                timestamp=time.time(),
                frame_sha1=None,
                action=event_details,
                mouse_xy=None,
                fps=None,
                notes=notes
            )

            with open(self.log_path, 'a') as f:
                f.write(log_entry.to_jsonl() + '\n')

            logger.debug(f"Logged event: {event_type}")

        except Exception as e:
            logger.error(f"Failed to log event: {e}")

    def get_log_stats(self) -> dict:
        """Get statistics about the log file.

        Returns:
            Dictionary with log file statistics
        """
        try:
            if not self.log_path.exists():
                return {"entries": 0, "size_bytes": 0}

            entries = 0
            with open(self.log_path) as f:
                for line in f:
                    if line.strip():
                        entries += 1

            size_bytes = self.log_path.stat().st_size

            return {
                "entries": entries,
                "size_bytes": size_bytes,
                "size_mb": size_bytes / (1024 * 1024)
            }

        except Exception as e:
            logger.error(f"Failed to get log stats: {e}")
            return {"entries": 0, "size_bytes": 0, "error": str(e)}


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Setup logging configuration.

    Args:
        level: Logging level
        log_file: Optional file for logging output
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logger.info(f"Logging configured: level={level}, file={log_file}")


def create_session_id() -> str:
    """Create unique session identifier.

    Returns:
        Unique session ID string
    """
    import datetime
    import random
    import string

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))

    return f"{timestamp}_{random_suffix}"


def get_log_filename(base_name: str = "agent_run") -> str:
    """Generate log filename with timestamp.

    Args:
        base_name: Base name for log file

    Returns:
        Log filename with timestamp
    """
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.jsonl"


def rotate_logs(log_dir: Path, max_files: int = 10, max_size_mb: int = 100) -> None:
    """Rotate log files to prevent disk space issues.

    Args:
        log_dir: Directory containing log files
        max_files: Maximum number of log files to keep
        max_size_mb: Maximum size per log file in MB
    """
    try:
        if not log_dir.exists():
            return

        # Find all JSONL log files
        log_files = list(log_dir.glob("*.jsonl"))
        log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)  # Newest first

        # Remove old files beyond max_files limit
        if len(log_files) > max_files:
            for old_file in log_files[max_files:]:
                logger.info(f"Removing old log file: {old_file}")
                old_file.unlink()

        # Check file sizes and warn about large files
        for log_file in log_files[:max_files]:
            size_mb = log_file.stat().st_size / (1024 * 1024)
            if size_mb > max_size_mb:
                logger.warning(f"Log file {log_file.name} is {size_mb:.1f}MB (>{max_size_mb}MB limit)")

        logger.debug(f"Log rotation complete: kept {min(len(log_files), max_files)} files")

    except Exception as e:
        logger.error(f"Log rotation failed: {e}")
