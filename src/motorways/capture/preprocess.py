"""Image preprocessing for model input preparation."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def prepare(
    obs_img: np.ndarray, input_size: tuple[int, int], normalize: bool = True
) -> np.ndarray:
    """Prepare observation image for model input.

    Args:
        obs_img: Input image array (H, W, 3) in RGB format
        input_size: Target size (width, height) for model input
        normalize: Whether to normalize pixel values to [0, 1]

    Returns:
        Preprocessed image array (1, C, H, W) ready for model inference

    Raises:
        ValueError: If input image format is invalid
    """
    if obs_img.ndim != 3 or obs_img.shape[2] != 3:
        raise ValueError(
            f"Expected RGB image with shape (H, W, 3), got {obs_img.shape}"
        )

    if obs_img.dtype != np.uint8:
        logger.warning(f"Input image dtype is {obs_img.dtype}, expected uint8")

    try:
        # Resize image to target size
        # Note: cv2.resize expects (width, height) but returns (height, width, channels)
        width, height = input_size
        resized = cv2.resize(obs_img, (width, height), interpolation=cv2.INTER_LINEAR)

        # Convert to float32 for model compatibility
        processed = resized.astype(np.float32)

        # Normalize pixel values if requested
        if normalize:
            processed = processed / 255.0

        # Convert from HWC to CHW format (channels first)
        processed = np.transpose(processed, (2, 0, 1))

        # Add batch dimension (1, C, H, W)
        processed = np.expand_dims(processed, axis=0)

        logger.debug(
            f"Preprocessed image: {obs_img.shape} -> {processed.shape}, "
            f"normalized={normalize}"
        )
        return processed

    except Exception as e:
        logger.error(f"Failed to preprocess image: {e}")
        raise


def resize_image(
    img: np.ndarray,
    target_size: tuple[int, int],
    interpolation: str = "linear",
) -> np.ndarray:
    """Resize image to target size.

    Args:
        img: Input image array (H, W, 3)
        target_size: Target (width, height)
        interpolation: Interpolation method ("linear", "nearest", "cubic", "area")

    Returns:
        Resized image array

    Raises:
        ValueError: If interpolation method is invalid
    """
    interpolation_map = {
        "linear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4
    }

    if interpolation not in interpolation_map:
        raise ValueError(f"Invalid interpolation method '{interpolation}'. Available: {list(interpolation_map.keys())}")

    width, height = target_size
    return cv2.resize(img, (width, height), interpolation=interpolation_map[interpolation])


def normalize_image(img: np.ndarray, method: str = "zero_one") -> np.ndarray:
    """Normalize image pixel values.

    Args:
        img: Input image array
        method: Normalization method ("zero_one", "mean_std", "imagenet")

    Returns:
        Normalized image array

    Raises:
        ValueError: If normalization method is invalid
    """
    if method == "zero_one":
        # Normalize to [0, 1]
        return img.astype(np.float32) / 255.0

    elif method == "mean_std":
        # Standardize to mean=0, std=1
        img_float = img.astype(np.float32) / 255.0
        mean = np.mean(img_float, axis=(0, 1), keepdims=True)
        std = np.std(img_float, axis=(0, 1), keepdims=True)
        return (img_float - mean) / (std + 1e-8)

    elif method == "imagenet":
        # ImageNet normalization
        img_float = img.astype(np.float32) / 255.0
        # ImageNet mean and std for RGB
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (img_float - mean) / std

    else:
        raise ValueError(f"Invalid normalization method '{method}'. Available: zero_one, mean_std, imagenet")


def hwc_to_chw(img: np.ndarray) -> np.ndarray:
    """Convert image from HWC to CHW format.

    Args:
        img: Image array in (H, W, C) format

    Returns:
        Image array in (C, H, W) format
    """
    if img.ndim != 3:
        raise ValueError(f"Expected 3D image array, got shape {img.shape}")

    return np.transpose(img, (2, 0, 1))


def add_batch_dimension(img: np.ndarray) -> np.ndarray:
    """Add batch dimension to image array.

    Args:
        img: Image array without batch dimension

    Returns:
        Image array with batch dimension as first axis
    """
    return np.expand_dims(img, axis=0)


def validate_image_format(img: np.ndarray, expected_shape: tuple[int, ...] = None) -> bool:
    """Validate image array format and properties.

    Args:
        img: Image array to validate
        expected_shape: Expected shape tuple (optional)

    Returns:
        True if image format is valid

    Raises:
        ValueError: If image format is invalid
    """
    if not isinstance(img, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(img)}")

    if img.size == 0:
        raise ValueError("Image array is empty")

    if expected_shape is not None and img.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {img.shape}")

    # Check for valid pixel values
    if img.dtype == np.uint8:
        if img.min() < 0 or img.max() > 255:
            raise ValueError(f"Invalid uint8 pixel values: range [{img.min()}, {img.max()}]")
    elif img.dtype == np.float32 or img.dtype == np.float64:
        # Allow flexible float ranges, but warn about unusual values
        if img.min() < -10 or img.max() > 10:
            logger.warning(f"Unusual float pixel value range: [{img.min():.3f}, {img.max():.3f}]")

    return True


def crop_center(img: np.ndarray, crop_size: tuple[int, int]) -> np.ndarray:
    """Crop center region of image.

    Args:
        img: Input image array (H, W, C)
        crop_size: Crop size (width, height)

    Returns:
        Center-cropped image

    Raises:
        ValueError: If crop size is larger than image
    """
    h, w = img.shape[:2]
    crop_w, crop_h = crop_size

    if crop_w > w or crop_h > h:
        raise ValueError(f"Crop size {crop_size} larger than image size ({w}, {h})")

    start_x = (w - crop_w) // 2
    start_y = (h - crop_h) // 2

    return img[start_y:start_y + crop_h, start_x:start_x + crop_w].copy()
