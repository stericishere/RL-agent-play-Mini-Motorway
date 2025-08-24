"""
Utility functions and configuration for reward system.

Provides common computer vision utilities, configuration management,
and helper functions used across different reward calculators.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path


@dataclass 
class RewardConfig:
    """
    Configuration for reward calculation systems.
    
    Provides centralized configuration for all reward calculators
    with sensible defaults that can be overridden.
    """
    
    # General settings
    max_history_frames: int = 10
    max_reward_history: int = 1000
    
    # Shaped reward weights
    survival_reward: float = 0.1
    construction_reward_scale: float = 1.0
    network_growth_scale: float = 2.0
    activity_reward_scale: float = 0.5
    
    # Traffic flow reward settings
    green_light_reward_scale: float = 2.0
    movement_reward_scale: float = 1.5
    congestion_penalty_scale: float = 3.0
    
    # Network progress reward settings  
    score_reward_scale: float = 10.0
    connection_reward_scale: float = 5.0
    resource_efficiency_scale: float = 1.0
    
    # Game over penalty
    game_over_penalty: float = -100.0
    
    # Computer vision parameters
    movement_threshold: float = 5.0  # Minimum pixel movement to count as motion
    green_light_hue_range: Tuple[int, int] = (50, 80)  # HSV hue range for green
    congestion_detection_threshold: float = 0.3  # Fraction of stationary pixels
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RewardConfig':
        """Create config from dictionary, using defaults for missing keys."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})


class CVUtils:
    """Computer vision utilities for reward calculation."""
    
    @staticmethod
    def detect_movement(current_frame: np.ndarray, previous_frame: np.ndarray, 
                       threshold: float = 5.0) -> Tuple[np.ndarray, float]:
        """
        Detect movement between two frames.
        
        Args:
            current_frame: Current RGB frame
            previous_frame: Previous RGB frame
            threshold: Minimum pixel difference to count as movement
            
        Returns:
            Tuple of (movement_mask, average_movement_magnitude)
        """
        if current_frame.shape != previous_frame.shape:
            raise ValueError("Frame shapes must match")
        
        # Convert to grayscale for movement detection
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_RGB2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(current_gray, previous_gray)
        
        # Apply threshold to get movement mask
        movement_mask = diff > threshold
        
        # Calculate average movement magnitude
        avg_movement = float(np.mean(diff[movement_mask])) if np.any(movement_mask) else 0.0
        
        return movement_mask, avg_movement
    
    @staticmethod
    def detect_green_lights(frame: np.ndarray, hue_range: Tuple[int, int] = (50, 80)) -> Tuple[int, np.ndarray]:
        """
        Detect green traffic lights in the frame.
        
        Args:
            frame: RGB frame
            hue_range: HSV hue range for green color detection
            
        Returns:
            Tuple of (count_of_green_lights, green_mask)
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        
        # Define green color range in HSV
        lower_green = np.array([hue_range[0], 50, 50])  # Lower bound
        upper_green = np.array([hue_range[1], 255, 255])  # Upper bound
        
        # Create mask for green colors
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Find contours to count individual lights
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size to avoid noise
        min_area = 10  # Minimum area for a traffic light
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        
        return len(valid_contours), green_mask
    
    @staticmethod
    def measure_visual_activity(current_frame: np.ndarray, previous_frame: Optional[np.ndarray] = None) -> float:
        """
        Measure overall visual activity in the frame.
        
        Uses edge detection and optionally frame differencing to measure
        how much is happening visually (cars moving, changes, etc.).
        
        Args:
            current_frame: Current RGB frame
            previous_frame: Optional previous frame for motion analysis
            
        Returns:
            Activity score (higher = more activity)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        
        # Detect edges using Canny
        edges = cv2.Canny(gray, 50, 150)
        edge_activity = np.sum(edges) / (gray.shape[0] * gray.shape[1])
        
        # If previous frame available, add motion component
        motion_activity = 0.0
        if previous_frame is not None:
            movement_mask, avg_movement = CVUtils.detect_movement(current_frame, previous_frame)
            motion_activity = avg_movement / 255.0  # Normalize to 0-1
        
        # Combine edge and motion activity
        total_activity = edge_activity * 0.6 + motion_activity * 0.4
        return float(total_activity)
    
    @staticmethod
    def detect_congestion(frame: np.ndarray, previous_frames: List[np.ndarray], 
                         threshold: float = 0.3) -> float:
        """
        Detect traffic congestion by analyzing stationary pixels over time.
        
        Args:
            frame: Current RGB frame
            previous_frames: List of previous frames for temporal analysis
            threshold: Fraction of pixels that must be stationary to indicate congestion
            
        Returns:
            Congestion score (0.0 = no congestion, 1.0 = full congestion)
        """
        if len(previous_frames) < 2:
            return 0.0  # Need history for congestion detection
        
        # Analyze movement over the last few frames
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        stationary_mask = np.ones_like(gray, dtype=bool)
        
        for prev_frame in previous_frames[-3:]:  # Use last 3 frames
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
            movement_mask, _ = CVUtils.detect_movement(frame, prev_frame, threshold=3.0)
            stationary_mask &= ~movement_mask  # Areas that haven't moved
        
        # Calculate fraction of stationary pixels
        stationary_fraction = np.sum(stationary_mask) / stationary_mask.size
        
        # Apply threshold and normalize
        if stationary_fraction > threshold:
            congestion_score = min(1.0, (stationary_fraction - threshold) / (1.0 - threshold))
        else:
            congestion_score = 0.0
            
        return float(congestion_score)
    
    @staticmethod
    def extract_score_region(frame: np.ndarray, score_region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Extract the score region from the frame for OCR.
        
        Args:
            frame: RGB frame
            score_region: Optional (x, y, width, height) region. If None, estimates location.
            
        Returns:
            Cropped region containing the score
        """
        if score_region is None:
            # Default to top-right corner (common score location)
            h, w = frame.shape[:2]
            x, y = int(w * 0.7), int(h * 0.05)  # Top right area
            width, height = int(w * 0.25), int(h * 0.15)  # Reasonable size
            score_region = (x, y, width, height)
        
        x, y, width, height = score_region
        
        # Ensure region is within frame bounds
        x = max(0, min(x, frame.shape[1] - 1))
        y = max(0, min(y, frame.shape[0] - 1))
        width = min(width, frame.shape[1] - x)
        height = min(height, frame.shape[0] - y)
        
        return frame[y:y+height, x:x+width]
    
    @staticmethod
    def preprocess_for_ocr(image_region: np.ndarray) -> np.ndarray:
        """
        Preprocess image region for better OCR results.
        
        Args:
            image_region: RGB image region containing text
            
        Returns:
            Preprocessed image optimized for OCR
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image_region, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding to handle varying lighting
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        
        # Scale up for better OCR (OCR works better on larger text)
        scale_factor = 3
        processed = cv2.resize(processed, None, fx=scale_factor, fy=scale_factor, 
                              interpolation=cv2.INTER_CUBIC)
        
        return processed