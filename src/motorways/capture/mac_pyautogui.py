"""macOS window capture using pyautogui as fallback."""

import logging
from typing import Optional, Tuple

import numpy as np
import pyautogui
from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGNullWindowID,
    kCGWindowListExcludeDesktopElements,
    kCGWindowListOptionOnScreenOnly,
)

logger = logging.getLogger(__name__)


def find_window_pyautogui(title_substr: str) -> tuple[Optional[int], Optional[dict]]:
    """Find window by title substring using pyautogui approach.

    Args:
        title_substr: Substring to search for in window titles

    Returns:
        Tuple of (window_id, bounds_dict) or (None, None) if not found
    """
    # Use the same window finding logic as the original
    window_list = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
        kCGNullWindowID
    )

    for window in window_list:
        window_name = window.get('kCGWindowName', '')
        if title_substr.lower() in window_name.lower():
            window_id = window.get('kCGWindowNumber')
            bounds = window.get('kCGWindowBounds', {})

            logger.info(f"Found window: {window_name} (ID: {window_id})")
            logger.info(f"Bounds: {bounds}")

            return window_id, bounds

    logger.warning(f"No window found containing '{title_substr}'")
    return None, None


def grab_window_region(bounds: dict) -> np.ndarray:
    """Capture window region using pyautogui screenshot.

    Args:
        bounds: Window bounds dictionary with X, Y, Width, Height

    Returns:
        RGB image array of shape (H, W, 3) with dtype uint8

    Raises:
        RuntimeError: If window capture fails
    """
    try:
        x = int(bounds['X'])
        y = int(bounds['Y'])
        width = int(bounds['Width'])
        height = int(bounds['Height'])
        
        logger.debug(f"Capturing region: {x}, {y}, {width}x{height}")
        
        # Use pyautogui to capture the screen region
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        
        # Convert PIL Image to numpy array
        rgb_array = np.array(screenshot)
        
        logger.debug(f"Captured region: {rgb_array.shape}")
        return rgb_array
        
    except Exception as e:
        logger.error(f"Failed to capture window region: {e}")
        raise RuntimeError(f"Window region capture failed: {e}") from e


def grab_window_pyautogui(window_id: int, bounds: dict) -> np.ndarray:
    """Capture window using pyautogui (fallback method).

    Args:
        window_id: Window ID (for compatibility, not used in this implementation)
        bounds: Window bounds dictionary

    Returns:
        RGB image array of shape (H, W, 3) with dtype uint8

    Raises:
        RuntimeError: If window capture fails
    """
    logger.info(f"Using pyautogui fallback for window {window_id}")
    return grab_window_region(bounds)