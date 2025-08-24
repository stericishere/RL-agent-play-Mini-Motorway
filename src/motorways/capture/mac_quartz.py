"""macOS window capture using CoreGraphics and PyObjC."""

import logging
from typing import Optional

import numpy as np
from Quartz import (
    CGWindowListCopyWindowInfo,
    CGWindowListCreateImage,
    kCGNullWindowID,
    kCGWindowImageBoundsIgnoreFraming,
    kCGWindowImageDefault,
    kCGWindowListExcludeDesktopElements,
    kCGWindowListOptionOnScreenOnly,
    kCGWindowListOptionAll,
)

logger = logging.getLogger(__name__)


def find_window(title_substr: str) -> tuple[Optional[int], Optional[dict]]:
    """Find window by title substring.

    Args:
        title_substr: Substring to search for in window titles

    Returns:
        Tuple of (window_id, bounds_dict) or (None, None) if not found
    """
    # First try on-screen windows only
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

    # If not found, try all windows (including fullscreen games)
    logger.debug("Searching in all windows (including fullscreen)")
    window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionAll, kCGNullWindowID)

    # Collect matching windows and prioritize by size and name
    matching_windows = []
    for window in window_list:
        window_name = window.get('kCGWindowName', '')
        owner_name = window.get('kCGWindowOwnerName', '')
        
        # Check both window name and owner name
        if (title_substr.lower() in window_name.lower() or 
            title_substr.lower() in owner_name.lower()):
            
            window_id = window.get('kCGWindowNumber')
            bounds = window.get('kCGWindowBounds', {})
            
            # Skip very small windows (likely UI elements)
            width = bounds.get('Width', 0)
            height = bounds.get('Height', 0)
            if width > 100 and height > 100:
                # Priority: named windows > larger windows
                priority = 0
                if window_name and len(window_name) > 0:
                    priority += 1000  # Named windows get high priority
                priority += width * height  # Larger windows get more priority
                
                matching_windows.append((priority, window_name, owner_name, window_id, bounds))
    
    # Return the highest priority window
    if matching_windows:
        matching_windows.sort(key=lambda x: x[0], reverse=True)
        _, window_name, owner_name, window_id, bounds = matching_windows[0]
        logger.info(f"Found window: {window_name} (Owner: {owner_name}, ID: {window_id})")
        logger.info(f"Bounds: {bounds}")
        return window_id, bounds

    logger.warning(f"No window found containing '{title_substr}'")
    return None, None


def grab_window(window_id: int, bounds: Optional[dict] = None) -> np.ndarray:
    """Capture window as RGB numpy array.

    Args:
        window_id: Window ID to capture
        bounds: Optional window bounds dict, fetched if not provided

    Returns:
        RGB image array of shape (H, W, 3) with dtype uint8

    Raises:
        RuntimeError: If window capture fails
    """
    # Get bounds if not provided
    if bounds is None:
        bounds = get_window_bounds(window_id)
        if bounds is None:
            raise RuntimeError(f"Could not get bounds for window {window_id}")
    
    try:
        # Create CGImage of the window
        cgimg = CGWindowListCreateImage(
            (0, 0, 0, 0),  # Capture entire window bounds
            kCGWindowListOptionOnScreenOnly,
            window_id,
            kCGWindowImageBoundsIgnoreFraming | kCGWindowImageDefault
        )

        if cgimg is None:
            raise RuntimeError(f"Failed to capture window {window_id}")

        # Use PIL to convert CGImage to numpy array (avoiding PyObjC issues)
        try:
            from PIL import Image
            import io
            
            # Convert CGImage to data using Core Graphics
            from Quartz import CGImageDestinationCreateWithData, CGImageDestinationAddImage, CGImageDestinationFinalize
            from CoreFoundation import CFDataCreateMutable, CFDataGetBytePtr, CFDataGetLength
            import ctypes
            
            # Create mutable data for PNG output
            data_ref = CFDataCreateMutable(None, 0)
            if not data_ref:
                raise RuntimeError("Failed to create mutable data")
            
            # Create image destination for PNG format
            image_dest = CGImageDestinationCreateWithData(data_ref, "public.png", 1, None)
            if not image_dest:
                raise RuntimeError("Failed to create image destination")
            
            # Add image to destination
            CGImageDestinationAddImage(image_dest, cgimg, None)
            
            # Finalize the image
            if not CGImageDestinationFinalize(image_dest):
                raise RuntimeError("Failed to finalize image")
            
            # Get the PNG data
            data_length = CFDataGetLength(data_ref)
            data_ptr = CFDataGetBytePtr(data_ref)
            
            # Convert to bytes
            buffer = (ctypes.c_ubyte * data_length).from_address(data_ptr)
            png_data = bytes(buffer)
            
            # Use PIL to decode PNG
            pil_image = Image.open(io.BytesIO(png_data))
            rgb_array = np.array(pil_image.convert('RGB'))
            
            logger.debug(f"Captured window {window_id}: {rgb_array.shape}")
            return rgb_array
            
        except ImportError:
            logger.error("PIL not available, trying fallback methods")
        except Exception as e:
            logger.error(f"PIL conversion failed: {e}, trying fallback methods")
        
        # Try pyautogui fallback before NSBitmapImageRep
        try:
            from .mac_pyautogui import grab_window_pyautogui
            logger.info(f"Falling back to pyautogui for window {window_id}")
            return grab_window_pyautogui(window_id, bounds)
        except Exception as e:
            logger.error(f"PyAutoGUI fallback failed: {e}")
        
        # Last resort: NSBitmapImageRep approach (problematic on some systems)
        logger.warning("Attempting NSBitmapImageRep as last resort")
        from Cocoa import NSBitmapImageRep
        
        rep = NSBitmapImageRep.alloc().initWithCGImage_(cgimg)
        if rep is None:
            raise RuntimeError("Failed to create NSBitmapImageRep")

        # Get pixel dimensions and format information
        width = int(rep.pixelsWide())
        height = int(rep.pixelsHigh())
        bytes_per_pixel = int(rep.bitsPerPixel()) // 8
        bytes_per_row = int(rep.bytesPerRow())
        
        logger.debug(f"Image info: {width}x{height}, {bytes_per_pixel} bytes/pixel, {bytes_per_row} bytes/row")

        if width == 0 or height == 0:
            raise RuntimeError("Captured image has zero dimensions")

        # More robust bitmap data handling
        try:
            # Try getting TIFF representation first (more reliable)
            tiff_data = rep.TIFFRepresentation()
            if tiff_data:
                try:
                    from PIL import Image
                    import io
                    pil_image = Image.open(io.BytesIO(bytes(tiff_data)))
                    return np.array(pil_image.convert('RGB'))
                except ImportError:
                    pass
                
            # Fallback to direct bitmap data access
            bitmap_data = rep.bitmapData()
            if bitmap_data is None:
                raise RuntimeError("Failed to get bitmap data")

            # Create numpy array from raw data with better error handling
            try:
                # Try np.frombuffer first
                buf = np.frombuffer(bitmap_data, dtype=np.uint8, count=height * bytes_per_row)
            except ValueError as e:
                logger.error(f"frombuffer failed: {e}, trying alternative method")
                # Alternative: use ctypes to extract data
                import ctypes
                expected_size = height * bytes_per_row
                buffer_type = ctypes.c_uint8 * expected_size
                buffer = buffer_type.from_address(int(bitmap_data))
                raw_data = bytes(buffer)
                buf = np.frombuffer(raw_data, dtype=np.uint8)

            # Handle different pixel formats
            if bytes_per_pixel == 4:  # RGBA or BGRA
                # Calculate actual pixels per row
                pixels_per_row = bytes_per_row // bytes_per_pixel
                try:
                    img_array = buf.reshape(height, pixels_per_row, 4)
                    # Crop to actual image width
                    img_array = img_array[:, :width, :]
                    # Convert BGRA to RGB (macOS typically uses BGRA)
                    rgb_arr = img_array[:, :, [2, 1, 0]].copy()
                except ValueError as e:
                    logger.error(f"Reshape failed: {e}")
                    # Try flattening and reshaping
                    total_pixels = len(buf) // 4
                    expected_pixels = width * height
                    if total_pixels >= expected_pixels:
                        img_array = buf[:expected_pixels * 4].reshape(height, width, 4)
                        rgb_arr = img_array[:, :, [2, 1, 0]].copy()
                    else:
                        raise RuntimeError(f"Insufficient pixel data: got {total_pixels}, need {expected_pixels}")
                        
            elif bytes_per_pixel == 3:  # RGB or BGR
                img_array = buf.reshape(height, width, 3)
                rgb_arr = img_array[:, :, ::-1].copy()  # BGR to RGB
            else:
                raise RuntimeError(f"Unsupported pixel format: {bytes_per_pixel} bytes per pixel")

            logger.debug(f"Captured window {window_id}: {rgb_arr.shape}")
            return rgb_arr
            
        except Exception as e:
            logger.error(f"Bitmap data extraction failed: {e}")
            raise RuntimeError(f"Unable to extract bitmap data: {e}")

    except Exception as e:
        logger.error(f"Failed to capture window {window_id}: {e}")
        raise RuntimeError(f"Window capture failed: {e}") from e


def get_window_bounds(window_id: int) -> Optional[dict]:
    """Get current window bounds.

    Args:
        window_id: Window ID to query

    Returns:
        Dictionary with X, Y, Width, Height keys or None if not found
    """
    window_list = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
        kCGNullWindowID
    )

    for window in window_list:
        if window.get('kCGWindowNumber') == window_id:
            return window.get('kCGWindowBounds', {})

    logger.warning(f"Window {window_id} not found")
    return None
