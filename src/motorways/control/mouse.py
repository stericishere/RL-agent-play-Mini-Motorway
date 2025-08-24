"""Mouse control wrappers using PyAutoGUI with safety and dry-run support."""

import logging
import time

import pyautogui

logger = logging.getLogger(__name__)

# Configure PyAutoGUI
pyautogui.PAUSE = 0.01  # Small pause between operations
pyautogui.FAILSAFE = True  # Enable failsafe


def click(x: int, y: int, dry_run: bool = False) -> None:
    """Click at screen coordinates.

    Args:
        x: Screen X coordinate
        y: Screen Y coordinate
        dry_run: If True, only log the action without clicking

    Raises:
        pyautogui.FailSafeException: If failsafe triggered
    """
    if dry_run:
        logger.info(f"DRY RUN: Click at ({x}, {y})")
        return

    try:
        # Move to position first to ensure accuracy
        pyautogui.moveTo(x, y, duration=0.01)
        time.sleep(0.01)  # Small delay for movement to complete
        pyautogui.click()

        logger.debug(f"Clicked at ({x}, {y})")

    except pyautogui.FailSafeException:
        logger.warning("PyAutoGUI failsafe triggered - mouse moved to corner")
        raise
    except Exception as e:
        logger.error(f"Failed to click at ({x}, {y}): {e}")
        raise


def drag_path(path: list[tuple[int, int]], dry_run: bool = False) -> None:
    """Drag mouse along a path of coordinates.

    Args:
        path: List of (x, y) coordinates to drag through
        dry_run: If True, only log the action without dragging

    Raises:
        ValueError: If path is empty or has less than 2 points
        pyautogui.FailSafeException: If failsafe triggered
    """
    if not path:
        raise ValueError("Drag path cannot be empty")

    if len(path) < 2:
        raise ValueError("Drag path must have at least 2 points")

    if dry_run:
        logger.info(f"DRY RUN: Drag from {path[0]} to {path[-1]} via {len(path)} points")
        return

    try:
        # Move to start position
        start_x, start_y = path[0]
        pyautogui.moveTo(start_x, start_y, duration=0.01)
        time.sleep(0.01)

        # Start drag
        pyautogui.mouseDown()

        # Drag through each point
        for i, (x, y) in enumerate(path[1:], 1):
            pyautogui.dragTo(x, y, duration=0.05)  # Slightly longer duration for smooth dragging

            # Small delay between points for reliability
            if i < len(path) - 1:
                time.sleep(0.01)

        # Release mouse
        pyautogui.mouseUp()

        logger.debug(f"Dragged from {path[0]} to {path[-1]} via {len(path)} points")

    except pyautogui.FailSafeException:
        logger.warning("PyAutoGUI failsafe triggered during drag - mouse moved to corner")
        # Ensure mouse is released
        try:
            pyautogui.mouseUp()
        except:
            pass
        raise
    except Exception as e:
        logger.error(f"Failed to drag path {path}: {e}")
        # Ensure mouse is released
        try:
            pyautogui.mouseUp()
        except:
            pass
        raise


def drag_line(start: tuple[int, int], end: tuple[int, int], dry_run: bool = False) -> None:
    """Drag from start to end coordinates.

    Args:
        start: Starting (x, y) coordinates
        end: Ending (x, y) coordinates
        dry_run: If True, only log the action without dragging

    Raises:
        pyautogui.FailSafeException: If failsafe triggered
    """
    if dry_run:
        logger.info(f"DRY RUN: Drag from {start} to {end}")
        return

    try:
        # Move to start position
        start_x, start_y = start
        end_x, end_y = end

        pyautogui.moveTo(start_x, start_y, duration=0.01)
        time.sleep(0.01)

        # Perform drag
        pyautogui.dragTo(end_x, end_y, duration=0.1)

        logger.debug(f"Dragged from {start} to {end}")

    except pyautogui.FailSafeException:
        logger.warning("PyAutoGUI failsafe triggered during line drag - mouse moved to corner")
        raise
    except Exception as e:
        logger.error(f"Failed to drag from {start} to {end}: {e}")
        raise


def get_mouse_position() -> tuple[int, int]:
    """Get current mouse position.

    Returns:
        Tuple of (x, y) current mouse coordinates
    """
    pos = pyautogui.position()
    return pos.x, pos.y


def is_failsafe_position() -> bool:
    """Check if mouse is in failsafe position (corner of screen).

    Returns:
        True if mouse is in failsafe position
    """
    try:
        x, y = get_mouse_position()
        screen_width, screen_height = pyautogui.size()

        # Check if mouse is in any corner (with small tolerance)
        tolerance = 5
        corners = [
            (0, 0),  # Top-left
            (screen_width - 1, 0),  # Top-right
            (0, screen_height - 1),  # Bottom-left
            (screen_width - 1, screen_height - 1),  # Bottom-right
        ]

        for corner_x, corner_y in corners:
            if abs(x - corner_x) <= tolerance and abs(y - corner_y) <= tolerance:
                return True

        return False

    except Exception:
        return False


def move_to_safe_position() -> None:
    """Move mouse away from failsafe corners to a safe position."""
    try:
        screen_width, screen_height = pyautogui.size()
        safe_x = screen_width // 2
        safe_y = screen_height // 2

        pyautogui.moveTo(safe_x, safe_y, duration=0.1)
        logger.debug(f"Moved mouse to safe position ({safe_x}, {safe_y})")

    except Exception as e:
        logger.error(f"Failed to move to safe position: {e}")


def configure_failsafe(enabled: bool = True) -> None:
    """Configure PyAutoGUI failsafe setting.

    Args:
        enabled: Whether to enable failsafe protection
    """
    pyautogui.FAILSAFE = enabled
    logger.info(f"PyAutoGUI failsafe {'enabled' if enabled else 'disabled'}")


def configure_pause(seconds: float = 0.01) -> None:
    """Configure PyAutoGUI pause between operations.

    Args:
        seconds: Pause duration in seconds
    """
    pyautogui.PAUSE = seconds
    logger.info(f"PyAutoGUI pause set to {seconds}s")
