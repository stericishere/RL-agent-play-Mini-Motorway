"""Tests for image preprocessing functionality."""

import pytest
import numpy as np

from motorways.capture.preprocess import (
    prepare, resize_image, normalize_image, hwc_to_chw, 
    add_batch_dimension, validate_image_format, crop_center
)


class TestPreprocessing:
    """Test image preprocessing functions."""
    
    def test_prepare_basic(self):
        """Test basic image preparation pipeline."""
        # Create test RGB image
        img = np.random.randint(0, 256, (100, 120, 3), dtype=np.uint8)
        
        result = prepare(img, (64, 64), normalize=True)
        
        # Check output format
        assert result.shape == (1, 3, 64, 64)
        assert result.dtype == np.float32
        assert 0 <= result.min() <= result.max() <= 1  # Normalized
    
    def test_prepare_without_normalization(self):
        """Test preparation without normalization."""
        img = np.random.randint(0, 256, (100, 120, 3), dtype=np.uint8)
        
        result = prepare(img, (32, 32), normalize=False)
        
        assert result.shape == (1, 3, 32, 32)
        assert result.dtype == np.float32
        assert 0 <= result.min() <= result.max() <= 255  # Not normalized
    
    def test_prepare_invalid_input(self):
        """Test error handling for invalid inputs."""
        # Wrong number of dimensions
        with pytest.raises(ValueError):
            prepare(np.ones((100, 100)), (64, 64))
        
        # Wrong number of channels
        with pytest.raises(ValueError):
            prepare(np.ones((100, 100, 4)), (64, 64))
        
        # Wrong channel count
        with pytest.raises(ValueError):
            prepare(np.ones((100, 100, 1)), (64, 64))
    
    def test_resize_image_basic(self):
        """Test basic image resizing."""
        img = np.random.randint(0, 256, (100, 120, 3), dtype=np.uint8)
        
        resized = resize_image(img, (64, 48))
        
        assert resized.shape == (48, 64, 3)  # Note: (H, W, C) format
        assert resized.dtype == np.uint8
    
    def test_resize_image_interpolation_methods(self):
        """Test different interpolation methods."""
        img = np.random.randint(0, 256, (100, 120, 3), dtype=np.uint8)
        
        # Test all supported interpolation methods
        methods = ["linear", "nearest", "cubic", "area"]
        
        for method in methods:
            resized = resize_image(img, (64, 48), method)
            assert resized.shape == (48, 64, 3)
    
    def test_resize_image_invalid_interpolation(self):
        """Test error handling for invalid interpolation method."""
        img = np.random.randint(0, 256, (100, 120, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError):
            resize_image(img, (64, 48), "invalid_method")
    
    def test_normalize_image_zero_one(self):
        """Test normalization to [0, 1] range."""
        img = np.array([[[0, 128, 255]]], dtype=np.uint8)  # 1x1x3 image
        
        normalized = normalize_image(img, "zero_one")
        
        assert normalized.dtype == np.float32
        np.testing.assert_allclose(normalized, [[[0.0, 128/255, 1.0]]], rtol=1e-6)
    
    def test_normalize_image_mean_std(self):
        """Test mean-std normalization."""
        img = np.ones((10, 10, 3), dtype=np.uint8) * 128
        
        normalized = normalize_image(img, "mean_std")
        
        assert normalized.dtype == np.float32
        # Should be close to zero mean
        assert abs(np.mean(normalized)) < 1e-6
    
    def test_normalize_image_invalid_method(self):
        """Test error handling for invalid normalization method."""
        img = np.ones((10, 10, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError):
            normalize_image(img, "invalid_method")
    
    def test_hwc_to_chw(self):
        """Test HWC to CHW format conversion."""
        img = np.random.rand(32, 48, 3).astype(np.float32)
        
        chw = hwc_to_chw(img)
        
        assert chw.shape == (3, 32, 48)
        
        # Check that data is preserved
        for c in range(3):
            np.testing.assert_array_equal(chw[c], img[:, :, c])
    
    def test_hwc_to_chw_invalid_input(self):
        """Test error handling for invalid input to HWC->CHW."""
        # Wrong number of dimensions
        with pytest.raises(ValueError):
            hwc_to_chw(np.ones((32, 48)))
    
    def test_add_batch_dimension(self):
        """Test adding batch dimension."""
        img = np.random.rand(3, 32, 48).astype(np.float32)
        
        batched = add_batch_dimension(img)
        
        assert batched.shape == (1, 3, 32, 48)
        np.testing.assert_array_equal(batched[0], img)
    
    def test_validate_image_format_valid(self):
        """Test validation of valid image formats."""
        img = np.random.randint(0, 256, (100, 120, 3), dtype=np.uint8)
        
        assert validate_image_format(img) is True
        assert validate_image_format(img, (100, 120, 3)) is True
    
    def test_validate_image_format_invalid_type(self):
        """Test validation error for non-numpy array."""
        with pytest.raises(ValueError):
            validate_image_format([1, 2, 3])
    
    def test_validate_image_format_empty(self):
        """Test validation error for empty array."""
        with pytest.raises(ValueError):
            validate_image_format(np.array([]))
    
    def test_validate_image_format_wrong_shape(self):
        """Test validation error for wrong shape."""
        img = np.ones((100, 120, 3))
        
        with pytest.raises(ValueError):
            validate_image_format(img, (50, 60, 3))
    
    def test_validate_image_format_invalid_pixel_values(self):
        """Test validation error for invalid pixel values."""
        # Invalid uint8 values
        img = np.ones((10, 10, 3), dtype=np.uint8) * 300  # Overflow
        
        with pytest.raises(ValueError):
            validate_image_format(img)
    
    def test_crop_center_basic(self):
        """Test center cropping."""
        img = np.random.randint(0, 256, (100, 120, 3), dtype=np.uint8)
        
        cropped = crop_center(img, (60, 80))
        
        assert cropped.shape == (80, 60, 3)  # Note: (H, W, C) format
    
    def test_crop_center_larger_than_image(self):
        """Test error when crop size is larger than image."""
        img = np.random.randint(0, 256, (50, 60, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError):
            crop_center(img, (100, 80))  # Larger than image
    
    def test_crop_center_exact_size(self):
        """Test cropping when crop size equals image size."""
        img = np.random.randint(0, 256, (50, 60, 3), dtype=np.uint8)
        
        cropped = crop_center(img, (60, 50))
        
        assert cropped.shape == (50, 60, 3)
        np.testing.assert_array_equal(cropped, img)


class TestPreprocessingIntegration:
    """Integration tests for preprocessing pipeline."""
    
    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        # Create test image with known properties
        img = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
        
        # Run full pipeline
        result = prepare(img, (128, 128), normalize=True)
        
        # Verify all transformations
        assert result.shape == (1, 3, 128, 128)
        assert result.dtype == np.float32
        assert 0 <= result.min() <= result.max() <= 1
        
        # Check that all channels have reasonable statistics
        for c in range(3):
            channel_data = result[0, c]
            assert channel_data.std() > 0  # Not all same value
    
    def test_preprocessing_deterministic(self):
        """Test that preprocessing is deterministic."""
        img = np.random.RandomState(42).randint(0, 256, (100, 120, 3), dtype=np.uint8)
        
        result1 = prepare(img, (64, 64), normalize=True)
        result2 = prepare(img, (64, 64), normalize=True)
        
        np.testing.assert_array_equal(result1, result2)
    
    def test_preprocessing_different_input_sizes(self):
        """Test preprocessing with various input sizes."""
        img = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        
        sizes = [(32, 32), (64, 64), (128, 128), (256, 256)]
        
        for w, h in sizes:
            result = prepare(img, (w, h), normalize=True)
            assert result.shape == (1, 3, h, w)
    
    @pytest.mark.parametrize("input_dtype", [np.uint8, np.float32, np.int32])
    def test_preprocessing_input_dtypes(self, input_dtype):
        """Test preprocessing with different input data types."""
        if input_dtype == np.uint8:
            img = np.random.randint(0, 256, (50, 60, 3), dtype=input_dtype)
        else:
            img = np.random.rand(50, 60, 3).astype(input_dtype)
            if input_dtype == np.int32:
                img = (img * 255).astype(input_dtype)
        
        result = prepare(img, (32, 32), normalize=True)
        
        assert result.shape == (1, 3, 32, 32)
        assert result.dtype == np.float32