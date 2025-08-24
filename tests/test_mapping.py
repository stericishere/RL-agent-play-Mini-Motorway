"""Tests for coordinate mapping functionality."""

import pytest
import numpy as np

from motorways.control.mapping import to_screen_center, crop_grid, validate_calibration
from motorways.config.schema import Calibration


class TestMapping:
    """Test coordinate mapping functions."""
    
    def test_to_screen_center_basic(self):
        """Test basic grid to screen coordinate conversion."""
        bounds = {'X': 100, 'Y': 100, 'Width': 1000, 'Height': 800}
        cal = Calibration(
            grid_h=10, grid_w=10,
            x0r=0.1, y0r=0.1, x1r=0.9, y1r=0.9
        )
        
        # Test corner cells
        x, y = to_screen_center(0, 0, bounds, cal)
        assert 180 <= x <= 190, f"Expected x around 185, got {x}"
        assert 140 <= y <= 150, f"Expected y around 145, got {y}"
        
        x, y = to_screen_center(9, 9, bounds, cal)
        assert 810 <= x <= 820, f"Expected x around 815, got {x}"
        assert 740 <= y <= 750, f"Expected y around 745, got {y}"
    
    def test_to_screen_center_out_of_bounds(self):
        """Test error handling for out-of-bounds coordinates."""
        bounds = {'X': 100, 'Y': 100, 'Width': 1000, 'Height': 800}
        cal = Calibration(
            grid_h=10, grid_w=10,
            x0r=0.1, y0r=0.1, x1r=0.9, y1r=0.9
        )
        
        with pytest.raises(ValueError):
            to_screen_center(-1, 0, bounds, cal)
        
        with pytest.raises(ValueError):
            to_screen_center(0, -1, bounds, cal)
        
        with pytest.raises(ValueError):
            to_screen_center(10, 0, bounds, cal)
        
        with pytest.raises(ValueError):
            to_screen_center(0, 10, bounds, cal)
    
    def test_crop_grid_basic(self):
        """Test image cropping functionality."""
        # Create test image
        img = np.ones((800, 1000, 3), dtype=np.uint8) * 128
        
        bounds = {'X': 0, 'Y': 0, 'Width': 1000, 'Height': 800}
        cal = Calibration(
            grid_h=10, grid_w=10,
            x0r=0.1, y0r=0.1, x1r=0.9, y1r=0.9
        )
        
        cropped = crop_grid(img, bounds, cal)
        
        # Check crop dimensions
        expected_h = int(0.8 * 800)  # y1r - y0r = 0.8
        expected_w = int(0.8 * 1000)  # x1r - x0r = 0.8
        
        assert cropped.shape[0] == expected_h
        assert cropped.shape[1] == expected_w
        assert cropped.shape[2] == 3
    
    def test_crop_grid_invalid_image(self):
        """Test error handling for invalid image formats."""
        bounds = {'X': 0, 'Y': 0, 'Width': 1000, 'Height': 800}
        cal = Calibration(
            grid_h=10, grid_w=10,
            x0r=0.1, y0r=0.1, x1r=0.9, y1r=0.9
        )
        
        # Wrong number of dimensions
        with pytest.raises(ValueError):
            crop_grid(np.ones((800, 1000)), bounds, cal)
        
        # Wrong number of channels
        with pytest.raises(ValueError):
            crop_grid(np.ones((800, 1000, 4)), bounds, cal)
    
    def test_validate_calibration_valid(self):
        """Test validation of valid calibration."""
        bounds = {'X': 100, 'Y': 100, 'Width': 1000, 'Height': 800}
        cal = Calibration(
            grid_h=10, grid_w=10,
            x0r=0.1, y0r=0.1, x1r=0.9, y1r=0.9
        )
        
        assert validate_calibration(cal, bounds) is True
    
    def test_validate_calibration_invalid_ratios(self):
        """Test validation of invalid ratio calibrations."""
        bounds = {'X': 100, 'Y': 100, 'Width': 1000, 'Height': 800}
        
        # x1r <= x0r
        cal = Calibration(
            grid_h=10, grid_w=10,
            x0r=0.9, y0r=0.1, x1r=0.1, y1r=0.9
        )
        assert validate_calibration(cal, bounds) is False
        
        # y1r <= y0r
        cal = Calibration(
            grid_h=10, grid_w=10,
            x0r=0.1, y0r=0.9, x1r=0.9, y1r=0.1
        )
        assert validate_calibration(cal, bounds) is False
    
    def test_validate_calibration_invalid_grid(self):
        """Test validation of invalid grid dimensions."""
        bounds = {'X': 100, 'Y': 100, 'Width': 1000, 'Height': 800}
        
        # Zero grid dimensions
        cal = Calibration(
            grid_h=0, grid_w=10,
            x0r=0.1, y0r=0.1, x1r=0.9, y1r=0.9
        )
        assert validate_calibration(cal, bounds) is False
        
        cal = Calibration(
            grid_h=10, grid_w=0,
            x0r=0.1, y0r=0.1, x1r=0.9, y1r=0.9
        )
        assert validate_calibration(cal, bounds) is False


@pytest.fixture
def sample_calibration():
    """Provide sample calibration for tests."""
    return Calibration(
        grid_h=16, grid_w=16,
        x0r=0.2, y0r=0.15, x1r=0.8, y1r=0.85,
        toolbar={"road": (0.1, 0.95), "bridge": (0.2, 0.95)}
    )


@pytest.fixture  
def sample_bounds():
    """Provide sample window bounds for tests."""
    return {'X': 200, 'Y': 50, 'Width': 1200, 'Height': 900}


class TestMappingIntegration:
    """Integration tests for mapping functions."""
    
    def test_grid_to_screen_roundtrip(self, sample_calibration, sample_bounds):
        """Test that grid coordinates map consistently."""
        cal = sample_calibration
        bounds = sample_bounds
        
        # Test multiple grid positions
        test_positions = [(0, 0), (0, 15), (15, 0), (15, 15), (7, 8)]
        
        for r, c in test_positions:
            x, y = to_screen_center(r, c, bounds, cal)
            
            # Coordinates should be within window bounds
            assert bounds['X'] <= x <= bounds['X'] + bounds['Width']
            assert bounds['Y'] <= y <= bounds['Y'] + bounds['Height']
            
            # Coordinates should be within grid region
            grid_x0 = bounds['X'] + cal.x0r * bounds['Width']
            grid_y0 = bounds['Y'] + cal.y0r * bounds['Height']
            grid_x1 = bounds['X'] + cal.x1r * bounds['Width']
            grid_y1 = bounds['Y'] + cal.y1r * bounds['Height']
            
            assert grid_x0 <= x <= grid_x1
            assert grid_y0 <= y <= grid_y1
    
    def test_crop_and_coordinates_consistency(self, sample_calibration, sample_bounds):
        """Test that cropping and coordinate mapping are consistent."""
        cal = sample_calibration
        bounds = sample_bounds
        
        # Create test image
        img = np.random.randint(0, 256, (bounds['Height'], bounds['Width'], 3), dtype=np.uint8)
        
        # Crop the grid region
        cropped = crop_grid(img, bounds, cal)
        
        # The cropped region should have reasonable dimensions
        expected_min_h = int(0.5 * bounds['Height'])
        expected_min_w = int(0.5 * bounds['Width'])
        
        assert cropped.shape[0] >= expected_min_h
        assert cropped.shape[1] >= expected_min_w
        assert cropped.shape[2] == 3