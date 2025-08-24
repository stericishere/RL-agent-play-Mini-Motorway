"""Coordinate mapping between grid and screen coordinates using calibration ratios."""

import logging

import numpy as np

from motorways.config.schema import Calibration

logger = logging.getLogger(__name__)


def to_screen_center(
    r: int, c: int, bounds: dict, cal: Calibration
) -> tuple[int, int]:
    """Convert grid coordinates to screen coordinates at cell center.

    Args:
        r: Grid row (0-based)
        c: Grid column (0-based)
        bounds: Window bounds dict with X, Y, Width, Height
        cal: Calibration with grid dimensions and ratio coordinates

    Returns:
        Tuple of (x, y) screen coordinates at cell center

    Raises:
        ValueError: If grid coordinates are out of bounds
    """
    if r < 0 or r >= cal.grid_h or c < 0 or c >= cal.grid_w:
        raise ValueError(f"Grid coordinates ({r}, {c}) out of bounds for {cal.grid_h}x{cal.grid_w} grid")

    # Calculate grid boundaries in screen coordinates
    x0 = bounds['X'] + cal.x0r * bounds['Width']
    y0 = bounds['Y'] + cal.y0r * bounds['Height']
    x1 = bounds['X'] + cal.x1r * bounds['Width']
    y1 = bounds['Y'] + cal.y1r * bounds['Height']

    # Calculate cell dimensions
    cell_width = (x1 - x0) / cal.grid_w
    cell_height = (y1 - y0) / cal.grid_h

    # Calculate center coordinates
    center_x = int(x0 + (c + 0.5) * cell_width)
    center_y = int(y0 + (r + 0.5) * cell_height)

    logger.debug(f"Grid ({r}, {c}) -> Screen ({center_x}, {center_y})")
    return center_x, center_y


def crop_grid(img: np.ndarray, bounds: dict, cal: Calibration) -> np.ndarray:
    """Crop image to show only the game grid region.

    Args:
        img: Full window image array (H, W, 3)
        bounds: Window bounds dict with X, Y, Width, Height
        cal: Calibration with grid boundary ratios

    Returns:
        Cropped image showing only grid region

    Raises:
        ValueError: If crop coordinates are invalid
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {img.shape}")

    height, width = img.shape[:2]

    # Calculate crop boundaries in image pixel coordinates
    # Note: bounds are in screen coordinates, need to map to image coordinates
    x0_ratio = cal.x0r
    y0_ratio = cal.y0r
    x1_ratio = cal.x1r
    y1_ratio = cal.y1r

    # Map ratios to image coordinates
    crop_x0 = int(x0_ratio * width)
    crop_y0 = int(y0_ratio * height)
    crop_x1 = int(x1_ratio * width)
    crop_y1 = int(y1_ratio * height)

    # Validate crop boundaries
    crop_x0 = max(0, min(crop_x0, width - 1))
    crop_y0 = max(0, min(crop_y0, height - 1))
    crop_x1 = max(crop_x0 + 1, min(crop_x1, width))
    crop_y1 = max(crop_y0 + 1, min(crop_y1, height))

    if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        raise ValueError(f"Invalid crop region: ({crop_x0}, {crop_y0}) to ({crop_x1}, {crop_y1})")

    # Crop the image
    grid_img = img[crop_y0:crop_y1, crop_x0:crop_x1].copy()

    logger.debug(f"Cropped from {img.shape} to {grid_img.shape} using ratios ({x0_ratio:.3f}, {y0_ratio:.3f}) to ({x1_ratio:.3f}, {y1_ratio:.3f})")
    return grid_img


def get_toolbar_coord(tool_name: str, bounds: dict, cal: Calibration) -> tuple[int, int]:
    """Get screen coordinates for toolbar button.

    Args:
        tool_name: Name of toolbar tool (e.g., "road", "bridge")
        bounds: Window bounds dict with X, Y, Width, Height
        cal: Calibration with toolbar coordinate ratios

    Returns:
        Tuple of (x, y) screen coordinates for toolbar button

    Raises:
        ValueError: If tool name not found in calibration
    """
    if tool_name not in cal.toolbar:
        raise ValueError(f"Tool '{tool_name}' not found in toolbar calibration. Available: {list(cal.toolbar.keys())}")

    tool_ratios = cal.toolbar[tool_name]
    x_ratio, y_ratio = tool_ratios

    # Convert ratios to screen coordinates
    screen_x = int(bounds['X'] + x_ratio * bounds['Width'])
    screen_y = int(bounds['Y'] + y_ratio * bounds['Height'])

    logger.debug(f"Toolbar '{tool_name}' -> Screen ({screen_x}, {screen_y})")
    return screen_x, screen_y


def get_upgrade_coord(choice: str, bounds: dict, cal: Calibration) -> tuple[int, int]:
    """Get screen coordinates for upgrade selection.

    Args:
        choice: Upgrade choice position ("left", "middle", "right")
        bounds: Window bounds dict with X, Y, Width, Height
        cal: Calibration with upgrade coordinates

    Returns:
        Tuple of (x, y) screen coordinates for upgrade choice

    Raises:
        ValueError: If upgrade choice not calibrated
    """
    if choice not in cal.upgrades:
        raise ValueError(f"Upgrade choice '{choice}' not calibrated")

    ux_ratio, uy_ratio = cal.upgrades[choice]
    
    # Convert ratio to screen coordinates
    upgrade_x = int(bounds['X'] + ux_ratio * bounds['Width'])
    upgrade_y = int(bounds['Y'] + uy_ratio * bounds['Height'])
    
    logger.debug(f"Upgrade '{choice}' -> Screen ({upgrade_x}, {upgrade_y})")
    return upgrade_x, upgrade_y


def detect_toolbar_buttons(img: np.ndarray, bounds: dict, cal: Calibration) -> list[tuple[str, int, int]]:
    """Detect toolbar buttons dynamically within the toolbar region.
    
    Args:
        img: Captured screen image
        bounds: Window bounds dict
        cal: Calibration with toolbar_region
        
    Returns:
        List of (button_name, x, y) tuples for detected buttons
        
    Note:
        This is a placeholder for future computer vision implementation.
        For now, falls back to fixed toolbar positions if available.
    """
    detected_buttons = []
    
    # Fallback to fixed positions if available
    for tool, (tx_ratio, ty_ratio) in cal.toolbar.items():
        toolbar_x = int(bounds['X'] + tx_ratio * bounds['Width'])
        toolbar_y = int(bounds['Y'] + ty_ratio * bounds['Height'])
        detected_buttons.append((tool, toolbar_x, toolbar_y))
    
    # TODO: Implement computer vision detection within cal.toolbar_region
    # This would analyze the image to find button positions dynamically
    
    logger.debug(f"Detected {len(detected_buttons)} toolbar buttons")
    return detected_buttons


def validate_calibration(cal: Calibration, bounds: dict) -> bool:
    """Validate calibration against window bounds.

    Args:
        cal: Calibration to validate
        bounds: Current window bounds

    Returns:
        True if calibration appears valid, False otherwise
    """
    try:
        # Check ratio bounds
        if not (0 <= cal.x0r < cal.x1r <= 1):
            logger.warning(f"Invalid X ratios: {cal.x0r} to {cal.x1r}")
            return False

        if not (0 <= cal.y0r < cal.y1r <= 1):
            logger.warning(f"Invalid Y ratios: {cal.y0r} to {cal.y1r}")
            return False

        # Check grid dimensions
        if cal.grid_h <= 0 or cal.grid_w <= 0:
            logger.warning(f"Invalid grid dimensions: {cal.grid_h}x{cal.grid_w}")
            return False

        # Test a few coordinate conversions
        for r in [0, cal.grid_h - 1]:
            for c in [0, cal.grid_w - 1]:
                x, y = to_screen_center(r, c, bounds, cal)
                if not (bounds['X'] <= x <= bounds['X'] + bounds['Width']):
                    logger.warning(f"Mapped X coordinate {x} outside window bounds")
                    return False
                if not (bounds['Y'] <= y <= bounds['Y'] + bounds['Height']):
                    logger.warning(f"Mapped Y coordinate {y} outside window bounds")
                    return False

        logger.info("Calibration validation passed")
        return True

    except Exception as e:
        logger.error(f"Calibration validation failed: {e}")
        return False
