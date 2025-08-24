"""Dynamic toolbar detection for Mini Motorways using computer vision."""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.distance import euclidean

from motorways.config.schema import Calibration

logger = logging.getLogger(__name__)


class ToolbarDetector:
    """Detects toolbar buttons dynamically as they change during gameplay."""
    
    def __init__(self):
        self.known_tools = {
            'road': {'color_hsv': (0, 0, 70), 'tolerance': 30},  # Gray roads
            'bridge': {'color_hsv': (25, 150, 200), 'tolerance': 30},  # Orange/brown bridges  
            'roundabout': {'color_hsv': (120, 100, 150), 'tolerance': 30},  # Green roundabouts
            'traffic_light': {'color_hsv': (0, 200, 200), 'tolerance': 30},  # Red traffic lights
            'highway': {'color_hsv': (220, 100, 180), 'tolerance': 30},  # Blue highways
            'tunnel': {'color_hsv': (300, 80, 120), 'tolerance': 30},  # Purple tunnels
        }
        self.button_cache = {}
        self.last_detection = None
    
    def detect_buttons(self, img: np.ndarray, bounds: dict, cal: Calibration) -> List[Tuple[str, int, int]]:
        """Detect toolbar buttons in the current frame.
        
        Args:
            img: Captured screen image
            bounds: Window bounds dict  
            cal: Calibration with toolbar_region
            
        Returns:
            List of (tool_name, x, y) tuples for detected buttons
        """
        if not cal.toolbar_region:
            logger.warning("No toolbar region calibrated, falling back to fixed positions")
            return self._fallback_to_fixed(bounds, cal)
        
        # Extract toolbar region from image
        toolbar_img = self._extract_toolbar_region(img, bounds, cal.toolbar_region)
        if toolbar_img is None:
            return self._fallback_to_fixed(bounds, cal)
        
        # Detect button positions
        buttons = self._find_button_positions(toolbar_img)
        
        # Convert to screen coordinates
        screen_buttons = []
        for tool_name, rel_x, rel_y in buttons:
            screen_x, screen_y = self._toolbar_to_screen_coords(
                rel_x, rel_y, bounds, cal.toolbar_region
            )
            screen_buttons.append((tool_name, screen_x, screen_y))
        
        # Cache the result
        self.last_detection = screen_buttons
        logger.debug(f"Detected {len(screen_buttons)} toolbar buttons: {[b[0] for b in screen_buttons]}")
        
        return screen_buttons
    
    def _extract_toolbar_region(self, img: np.ndarray, bounds: dict, toolbar_region: Dict[str, float]) -> Optional[np.ndarray]:
        """Extract the toolbar region from the full image."""
        try:
            height, width = img.shape[:2]
            
            # Convert ratios to pixel coordinates
            x0 = int(toolbar_region['x0r'] * width)
            y0 = int(toolbar_region['y0r'] * height)
            x1 = int(toolbar_region['x1r'] * width)
            y1 = int(toolbar_region['y1r'] * height)
            
            # Validate bounds
            x0 = max(0, min(x0, width - 1))
            y0 = max(0, min(y0, height - 1))
            x1 = max(x0 + 1, min(x1, width))
            y1 = max(y0 + 1, min(y1, height))
            
            # Extract region
            toolbar_img = img[y0:y1, x0:x1].copy()
            logger.debug(f"Extracted toolbar region: {toolbar_img.shape}")
            
            return toolbar_img
            
        except Exception as e:
            logger.error(f"Failed to extract toolbar region: {e}")
            return None
    
    def _find_button_positions(self, toolbar_img: np.ndarray) -> List[Tuple[str, int, int]]:
        """Find button positions within the toolbar image using computer vision."""
        buttons = []
        
        # Convert to HSV for better color detection
        hsv_img = cv2.cvtColor(toolbar_img, cv2.COLOR_RGB2HSV)
        
        # Look for circular/rectangular button shapes
        button_candidates = self._detect_button_shapes(toolbar_img)
        
        # Match candidates to known tools by color/pattern
        for candidate_x, candidate_y, candidate_region in button_candidates:
            tool_name = self._classify_button(candidate_region, hsv_img[
                candidate_y-10:candidate_y+10, 
                candidate_x-10:candidate_x+10
            ] if self._in_bounds(candidate_x, candidate_y, hsv_img, 10) else hsv_img)
            
            if tool_name:
                buttons.append((tool_name, candidate_x, candidate_y))
        
        # Sort buttons left-to-right (typical toolbar layout)
        buttons.sort(key=lambda b: b[1])
        
        return buttons
    
    def _detect_button_shapes(self, toolbar_img: np.ndarray) -> List[Tuple[int, int, np.ndarray]]:
        """Detect button-like shapes in the toolbar image."""
        candidates = []
        
        # Convert to grayscale for shape detection
        gray = cv2.cvtColor(toolbar_img, cv2.COLOR_RGB2GRAY)
        
        # Use edge detection to find button boundaries
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours (potential button shapes)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter by size - buttons should be reasonably sized
            area = cv2.contourArea(contour)
            if 200 < area < 5000:  # Adjust based on actual button sizes
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio - buttons are roughly square/rectangular
                aspect_ratio = w / h if h > 0 else 0
                if 0.5 < aspect_ratio < 2.0:
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Extract the button region for classification
                    button_region = toolbar_img[y:y+h, x:x+w]
                    candidates.append((center_x, center_y, button_region))
        
        logger.debug(f"Found {len(candidates)} button candidates")
        return candidates
    
    def _classify_button(self, button_region: np.ndarray, hsv_region: np.ndarray) -> Optional[str]:
        """Classify a button region to determine which tool it represents."""
        if button_region.size == 0 or hsv_region.size == 0:
            return None
        
        # Get dominant colors in the button region
        dominant_color = self._get_dominant_hsv_color(hsv_region)
        
        # Match against known tool colors
        best_match = None
        best_distance = float('inf')
        
        for tool_name, tool_info in self.known_tools.items():
            expected_hsv = tool_info['color_hsv']
            tolerance = tool_info['tolerance']
            
            # Calculate color distance
            distance = euclidean(dominant_color, expected_hsv)
            
            if distance < tolerance and distance < best_distance:
                best_distance = distance
                best_match = tool_name
        
        if best_match:
            logger.debug(f"Classified button as {best_match} (distance: {best_distance:.1f})")
        
        return best_match
    
    def _get_dominant_hsv_color(self, hsv_region: np.ndarray) -> Tuple[float, float, float]:
        """Get the dominant HSV color in a region."""
        # Flatten the region and get mean color
        if hsv_region.size == 0:
            return (0, 0, 0)
        
        reshaped = hsv_region.reshape(-1, hsv_region.shape[-1])
        mean_color = np.mean(reshaped, axis=0)
        
        return tuple(mean_color)
    
    def _in_bounds(self, x: int, y: int, img: np.ndarray, margin: int) -> bool:
        """Check if coordinates with margin are within image bounds."""
        height, width = img.shape[:2]
        return (margin <= x < width - margin and 
                margin <= y < height - margin)
    
    def _toolbar_to_screen_coords(self, rel_x: int, rel_y: int, bounds: dict, 
                                 toolbar_region: Dict[str, float]) -> Tuple[int, int]:
        """Convert toolbar-relative coordinates to screen coordinates."""
        # Get toolbar region in screen coordinates
        toolbar_x0 = bounds['X'] + toolbar_region['x0r'] * bounds['Width']
        toolbar_y0 = bounds['Y'] + toolbar_region['y0r'] * bounds['Height']
        
        # Convert relative position to screen position
        screen_x = int(toolbar_x0 + rel_x)
        screen_y = int(toolbar_y0 + rel_y)
        
        return screen_x, screen_y
    
    def _fallback_to_fixed(self, bounds: dict, cal: Calibration) -> List[Tuple[str, int, int]]:
        """Fallback to fixed toolbar positions if dynamic detection fails."""
        buttons = []
        
        for tool, (tx_ratio, ty_ratio) in cal.toolbar.items():
            toolbar_x = int(bounds['X'] + tx_ratio * bounds['Width'])
            toolbar_y = int(bounds['Y'] + ty_ratio * bounds['Height'])
            buttons.append((tool, toolbar_x, toolbar_y))
        
        logger.debug(f"Using fallback fixed positions for {len(buttons)} buttons")
        return buttons
    
    def get_available_tools(self, img: np.ndarray, bounds: dict, cal: Calibration) -> List[str]:
        """Get list of currently available tools."""
        detected_buttons = self.detect_buttons(img, bounds, cal)
        return [tool_name for tool_name, _, _ in detected_buttons]
    
    def find_tool_position(self, tool_name: str, img: np.ndarray, bounds: dict, 
                          cal: Calibration) -> Optional[Tuple[int, int]]:
        """Find the screen position of a specific tool."""
        detected_buttons = self.detect_buttons(img, bounds, cal)
        
        for detected_tool, x, y in detected_buttons:
            if detected_tool == tool_name:
                return (x, y)
        
        logger.warning(f"Tool '{tool_name}' not found in toolbar")
        return None


# Global instance for reuse
_toolbar_detector = ToolbarDetector()


def detect_toolbar_buttons(img: np.ndarray, bounds: dict, cal: Calibration) -> List[Tuple[str, int, int]]:
    """Detect toolbar buttons dynamically - main interface function."""
    return _toolbar_detector.detect_buttons(img, bounds, cal)


def get_available_tools(img: np.ndarray, bounds: dict, cal: Calibration) -> List[str]:
    """Get list of currently available tools."""
    return _toolbar_detector.get_available_tools(img, bounds, cal)


def find_tool_position(tool_name: str, img: np.ndarray, bounds: dict, cal: Calibration) -> Optional[Tuple[int, int]]:
    """Find the screen position of a specific tool."""
    return _toolbar_detector.find_tool_position(tool_name, img, bounds, cal)