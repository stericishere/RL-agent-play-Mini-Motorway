"""
Shaped reward calculator for Mini Motorways RL training.

Implements dense, heuristic-based rewards to provide continuous feedback
during training. This is the Phase 1 implementation focusing on basic
but effective reward signals that can be calculated without complex CV.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional
from .base import BaseRewardCalculator, RewardResult
from .utils import RewardConfig, CVUtils
from ..policy.action_space import Action


class ShapedRewardCalculator(BaseRewardCalculator):
    """
    Shaped reward calculator providing dense feedback through heuristics.
    
    This calculator provides immediate, dense reward signals to guide RL training
    even when the game score changes are sparse. It focuses on behaviors that
    generally lead to good gameplay without requiring complex computer vision.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize shaped reward calculator.
        
        Args:
            config: Configuration dictionary for reward parameters
        """
        super().__init__(config)
        
        # Use RewardConfig for consistent parameter management
        if isinstance(config, dict):
            self.reward_config = RewardConfig.from_dict(config)
        else:
            self.reward_config = config if config else RewardConfig()
        
        # State tracking for temporal rewards
        self._last_activity_score = 0.0
        self._construction_count = 0
        self._network_size_history = []
    
    def calculate_reward(
        self,
        current_obs: np.ndarray,
        previous_obs: Optional[np.ndarray],
        action: Optional[Action],
        info: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """
        Calculate shaped reward from multiple heuristic components.
        
        Args:
            current_obs: Current observation (RGB image array)
            previous_obs: Previous observation (None for first step)
            action: Action taken to reach current state (None for first step)
            info: Additional context information
            
        Returns:
            RewardResult containing total reward and component breakdown
        """
        components = {}
        metadata = {}
        
        # 1. Survival Reward - Basic reward for staying alive
        components['survival'] = self.reward_config.survival_reward
        
        # 2. Construction Quality Reward - Reward good building placement
        if action and action.type in ['click', 'drag']:
            construction_reward = self._evaluate_construction_quality(action, current_obs, previous_obs)
            components['construction'] = construction_reward * self.reward_config.construction_reward_scale
            self._construction_count += 1
        else:
            components['construction'] = 0.0
        
        # 3. Network Growth Reward - Encourage expanding the road network
        if previous_obs is not None:
            network_growth = self._measure_network_growth(current_obs, previous_obs)
            components['network_growth'] = network_growth * self.reward_config.network_growth_scale
        else:
            components['network_growth'] = 0.0
        
        # 4. Activity Reward - Reward visual activity (cars moving, etc.)
        activity_reward = self._measure_activity_reward(current_obs, previous_obs)
        components['activity'] = activity_reward * self.reward_config.activity_reward_scale
        
        # 5. Efficiency Penalty - Penalize wasteful actions
        efficiency_penalty = self._calculate_efficiency_penalty(action, current_obs)
        components['efficiency'] = efficiency_penalty
        
        # 6. Game Over Detection - Large penalty for losing
        game_over = self._detect_game_over(current_obs, previous_obs)
        if game_over:
            components['game_over'] = self.reward_config.game_over_penalty
        else:
            components['game_over'] = 0.0
        
        # Calculate total reward
        total_reward = sum(components.values())
        
        # Store metadata for analysis
        metadata = {
            'step_count': self.step_count,
            'construction_count': self._construction_count,
            'network_size_history_length': len(self._network_size_history),
            'last_activity_score': self._last_activity_score,
            'action_type': action.type if action else None,
        }
        
        result = RewardResult(
            total_reward=total_reward,
            components=components,
            metadata=metadata,
            game_over=game_over
        )
        
        self._store_step_data(current_obs, result)
        return result
    
    def _evaluate_construction_quality(self, action: Action, current_obs: np.ndarray, 
                                     previous_obs: Optional[np.ndarray]) -> float:
        """
        Evaluate the quality of a construction action.
        
        Provides positive reward for actions that seem constructive and
        negative reward for actions that seem wasteful or poorly placed.
        
        Args:
            action: The construction action taken
            current_obs: Current observation
            previous_obs: Previous observation
            
        Returns:
            Construction quality score
        """
        if previous_obs is None:
            return 0.5  # Neutral reward for first action
        
        # Basic heuristics for construction quality
        quality_score = 0.0
        
        # 1. Reward building in areas with less existing infrastructure
        # (encourages expansion rather than overbuilding)
        if hasattr(action, 'r') and hasattr(action, 'c'):
            # Extract region around the action for analysis
            h, w = current_obs.shape[:2]
            grid_h, grid_w = 32, 32  # From action space configuration
            
            # Convert grid coordinates to pixel coordinates
            pixel_r = int((action.r / grid_h) * h)
            pixel_c = int((action.c / grid_w) * w)
            
            # Sample region around the action (10% of image size)
            region_size = max(h, w) // 10
            r_start = max(0, pixel_r - region_size // 2)
            r_end = min(h, pixel_r + region_size // 2)
            c_start = max(0, pixel_c - region_size // 2)
            c_end = min(w, pixel_c + region_size // 2)
            
            # Analyze density of existing infrastructure in region
            current_region = current_obs[r_start:r_end, c_start:c_end]
            previous_region = previous_obs[r_start:r_end, c_start:c_end]
            
            # Check for changes (indicating construction)
            if current_region.shape == previous_region.shape:
                region_change = np.mean(np.abs(current_region.astype(float) - previous_region.astype(float)))
                
                # Reward moderate change (indicates construction)
                if 5.0 < region_change < 50.0:
                    quality_score += 1.0
                elif region_change > 50.0:
                    # Large changes might indicate overbuilding or poor placement
                    quality_score -= 0.5
        
        # 2. Reward drag actions more than clicks (roads are generally good)
        if action.type == 'drag':
            quality_score += 0.3
        elif action.type == 'click':
            # Clicks can be buildings or tools, context-dependent
            quality_score += 0.1
        
        # 3. Penalize actions that don't seem to change anything
        if previous_obs is not None:
            overall_change = np.mean(np.abs(current_obs.astype(float) - previous_obs.astype(float)))
            if overall_change < 1.0:  # Very little change
                quality_score -= 0.5
        
        return quality_score
    
    def _measure_network_growth(self, current_obs: np.ndarray, previous_obs: np.ndarray) -> float:
        """
        Measure growth in the road network size.
        
        Uses edge detection to estimate the amount of linear infrastructure
        (roads) in the image and rewards increases.
        
        Args:
            current_obs: Current observation
            previous_obs: Previous observation
            
        Returns:
            Network growth reward (positive for growth, negative for reduction)
        """
        # Convert to grayscale for edge detection
        current_gray = cv2.cvtColor(current_obs, cv2.COLOR_RGB2GRAY) if len(current_obs.shape) == 3 else current_obs
        previous_gray = cv2.cvtColor(previous_obs, cv2.COLOR_RGB2GRAY) if len(previous_obs.shape) == 3 else previous_obs
        
        # Use Canny edge detection to find linear features (roads)
        current_edges = cv2.Canny(current_gray, 50, 150)
        previous_edges = cv2.Canny(previous_gray, 50, 150)
        
        # Count edge pixels as a proxy for network size
        current_network_size = np.sum(current_edges > 0)
        previous_network_size = np.sum(previous_edges > 0)
        
        # Store in history for trend analysis
        self._network_size_history.append(current_network_size)
        if len(self._network_size_history) > 10:
            self._network_size_history.pop(0)
        
        # Calculate growth
        growth = current_network_size - previous_network_size
        
        # Normalize by image size
        image_size = current_obs.shape[0] * current_obs.shape[1]
        normalized_growth = growth / image_size
        
        # Apply scaling - reward growth, penalize shrinkage
        if normalized_growth > 0:
            return min(1.0, normalized_growth * 1000)  # Cap at 1.0
        else:
            return max(-0.5, normalized_growth * 500)  # Limit penalty
    
    def _measure_activity_reward(self, current_obs: np.ndarray, previous_obs: Optional[np.ndarray]) -> float:
        """
        Measure visual activity as a proxy for traffic flow.
        
        Rewards active, dynamic scenes which typically indicate good traffic flow.
        
        Args:
            current_obs: Current observation
            previous_obs: Previous observation
            
        Returns:
            Activity reward score
        """
        activity_score = CVUtils.measure_visual_activity(current_obs, previous_obs)
        
        # Track activity over time
        activity_change = activity_score - self._last_activity_score
        self._last_activity_score = activity_score
        
        # Reward moderate activity (traffic flowing) 
        # Penalize very low activity (gridlock) or very high activity (chaos)
        if 0.1 < activity_score < 0.8:
            activity_reward = activity_score
        elif activity_score <= 0.1:
            activity_reward = -0.3  # Penalty for no activity
        else:
            activity_reward = 0.8 - (activity_score - 0.8)  # Decreasing reward for excessive activity
        
        # Small bonus for increasing activity
        if activity_change > 0:
            activity_reward += activity_change * 0.1
        
        return activity_reward
    
    def _calculate_efficiency_penalty(self, action: Optional[Action], current_obs: np.ndarray) -> float:
        """
        Calculate penalty for inefficient actions.
        
        Penalizes actions that seem wasteful or counterproductive.
        
        Args:
            action: Action taken
            current_obs: Current observation
            
        Returns:
            Efficiency penalty (negative values are penalties)
        """
        if action is None:
            return 0.0
        
        penalty = 0.0
        
        # Penalize excessive clicking in the same general area
        if hasattr(action, 'r') and hasattr(action, 'c'):
            # Check if recent actions were in similar locations
            recent_actions = [r.metadata.get('action_type') for r in self._reward_history[-5:]]
            if len(recent_actions) >= 3 and all(a == action.type for a in recent_actions):
                penalty -= 0.2  # Small penalty for repetitive actions
        
        # Penalize no-op actions after several steps
        if action.type == 'noop' and self.step_count > 10:
            penalty -= 0.1
        
        return penalty
    
    def _detect_game_over(self, current_obs: np.ndarray, previous_obs: Optional[np.ndarray]) -> bool:
        """
        Detect game over conditions using heuristics.
        
        This is a basic implementation that will be improved in later phases
        with more sophisticated computer vision.
        
        Args:
            current_obs: Current observation
            previous_obs: Previous observation
            
        Returns:
            True if game over detected, False otherwise
        """
        # Simple heuristic: if activity has been extremely low for several steps,
        # it might indicate a gridlock/game over situation
        if len(self._reward_history) < 10:
            return False
        
        recent_activities = [
            r.components.get('activity', 0.0) 
            for r in self._reward_history[-10:]
        ]
        
        avg_recent_activity = np.mean(recent_activities)
        
        # Game over if activity has been very low for extended period
        if avg_recent_activity < 0.05 and self.step_count > 50:
            return True
        
        # Additional heuristic: if we haven't seen any construction
        # for a long time and activity is low, might be stuck
        recent_construction = sum(
            1 for r in self._reward_history[-20:] 
            if r.components.get('construction', 0.0) > 0
        )
        
        if recent_construction == 0 and avg_recent_activity < 0.1 and self.step_count > 100:
            return True
        
        return False
    
    def reset(self):
        """Reset calculator state for new episode."""
        super().reset()
        self._last_activity_score = 0.0
        self._construction_count = 0
        self._network_size_history.clear()