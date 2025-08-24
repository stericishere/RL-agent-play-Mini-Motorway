"""
Gymnasium environment wrapper for Mini Motorways RL training.

Provides a standard Gymnasium interface for training RL agents on Mini Motorways,
integrating the reward system with the existing screen capture and control infrastructure.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple
import time
import logging
from pathlib import Path

from .capture.mac_quartz import find_window, grab_window
from .capture.preprocess import prepare
from .config.schema import Calibration
from .control.mapping import to_screen_center, get_toolbar_coord, validate_calibration
from .control.mouse import click, drag_path
from .policy.action_space import Action, decode_action, get_action_space_size
from .rewards import ShapedRewardCalculator, RewardConfig
from .rewards.base import BaseRewardCalculator, RewardResult

logger = logging.getLogger(__name__)


class MiniMotorwaysEnv(gym.Env):
    """
    Gymnasium environment for Mini Motorways RL training.
    
    Provides a standard Gymnasium interface that integrates:
    - Screen capture and preprocessing
    - Action execution (mouse clicks/drags)  
    - Reward calculation using the reward system
    - Episode management and game state tracking
    
    The environment captures the game screen, processes actions through
    the mouse control system, and calculates rewards using the reward system.
    """
    
    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}
    
    def __init__(
        self,
        window_title: str = "Mini Motorways",
        calibration_path: Optional[Path] = None,
        reward_calculator: Optional[BaseRewardCalculator] = None,
        reward_config: Optional[Dict[str, Any]] = None,
        input_size: Tuple[int, int] = (128, 128),
        grid_size: Tuple[int, int] = (32, 32),
        max_episode_steps: int = 1000,
        capture_timeout: float = 5.0,
        dry_run: bool = False,
    ):
        """
        Initialize the Mini Motorways environment.
        
        Args:
            window_title: Substring to search for in window titles
            calibration_path: Path to calibration file
            reward_calculator: Reward calculator to use (default: ShapedRewardCalculator)
            reward_config: Configuration for reward calculation
            input_size: Size to resize captured frames to
            grid_size: Grid dimensions for action space
            max_episode_steps: Maximum steps per episode
            capture_timeout: Timeout for window capture operations
            dry_run: If True, don't actually execute mouse actions
        """
        super().__init__()
        
        # Configuration
        self.window_title = window_title
        self.calibration_path = calibration_path
        self.input_size = input_size
        self.grid_size = grid_size
        self.max_episode_steps = max_episode_steps
        self.capture_timeout = capture_timeout
        self.dry_run = dry_run
        
        # Initialize reward system
        if reward_calculator is None:
            reward_config = reward_config or {}
            if isinstance(reward_config, dict):
                reward_config = RewardConfig.from_dict(reward_config)
            self.reward_calculator = ShapedRewardCalculator(reward_config)
        else:
            self.reward_calculator = reward_calculator
        
        # Action and observation spaces
        action_space_size = get_action_space_size(grid_size[0], grid_size[1])
        self.action_space = spaces.Discrete(action_space_size)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(*input_size, 3),
            dtype=np.float32
        )
        
        # Episode state
        self.current_step = 0
        self.window_id = None
        self.window_bounds = None
        self.calibration = None
        self.last_observation = None
        self.episode_rewards = []
        
        # Load calibration if provided
        if calibration_path and calibration_path.exists():
            try:
                self.calibration = Calibration.load(calibration_path)
                logger.info(f"Loaded calibration from {calibration_path}")
            except Exception as e:
                logger.error(f"Failed to load calibration: {e}")
                raise
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Tuple of (initial_observation, info_dict)
        """
        super().reset(seed=seed)
        
        # Reset episode state
        self.current_step = 0
        self.episode_rewards = []
        self.last_observation = None
        
        # Reset reward calculator
        self.reward_calculator.reset()
        
        # Find and validate game window
        self.window_id, self.window_bounds = find_window(self.window_title)
        if self.window_id is None or self.window_bounds is None:
            raise RuntimeError(f"Could not find window containing '{self.window_title}'")
        
        # Validate calibration against current window
        if self.calibration and not validate_calibration(self.calibration, self.window_bounds):
            logger.warning("Calibration validation failed - may affect action accuracy")
        
        # Capture initial observation
        initial_obs = self._capture_observation()
        
        info = {
            'window_id': self.window_id,
            'window_bounds': self.window_bounds,
            'episode_step': self.current_step,
            'reward_components': {},
        }
        
        logger.info(f"Environment reset - Window ID: {self.window_id}, Bounds: {self.window_bounds}")
        return initial_obs, info
    
    def step(self, action_value: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action_value: Discrete action value to execute
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.window_id is None:
            raise RuntimeError("Environment not properly initialized. Call reset() first.")
        
        self.current_step += 1
        
        # Decode and execute action
        action = decode_action(action_value, self.grid_size[0], self.grid_size[1])
        execution_success = self._execute_action(action)
        
        # Capture new observation
        current_obs = self._capture_observation()
        
        # Calculate reward
        reward_result = self.reward_calculator.calculate_reward(
            current_obs, self.last_observation, action,
            info={'execution_success': execution_success}
        )
        
        reward = reward_result.total_reward
        self.episode_rewards.append(reward)
        
        # Check termination conditions
        terminated = reward_result.game_over
        truncated = self.current_step >= self.max_episode_steps
        
        # Prepare info dictionary
        info = {
            'episode_step': self.current_step,
            'action_type': action.type,
            'execution_success': execution_success,
            'reward_components': reward_result.components,
            'reward_metadata': reward_result.metadata,
            'episode_rewards_sum': sum(self.episode_rewards),
            'episode_rewards_mean': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
        }
        
        # Store observation for next step
        self.last_observation = current_obs
        
        logger.debug(
            f"Step {self.current_step}: Action={action.type}, "
            f"Reward={reward:.3f}, Terminated={terminated}, Truncated={truncated}"
        )
        
        return current_obs, reward, terminated, truncated, info
    
    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Render mode ("rgb_array" supported)
            
        Returns:
            RGB array of current screen capture or None
        """
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        
        if self.window_id is None:
            return None
        
        try:
            # Capture current screen
            raw_frame = grab_window(self.window_id, self.window_bounds)
            if raw_frame is None:
                return None
            
            # Process frame similar to observation
            processed = prepare(raw_frame, self.input_size, normalize=False)
            
            # Convert back to uint8 for rendering
            if processed.dtype == np.float32:
                processed = (processed * 255).astype(np.uint8)
            
            return processed
            
        except Exception as e:
            logger.error(f"Render failed: {e}")
            return None
    
    def close(self):
        """Clean up environment resources."""
        logger.info("Environment closed")
        
        # Print episode summary if we have reward data
        if self.episode_rewards:
            total_reward = sum(self.episode_rewards)
            mean_reward = np.mean(self.episode_rewards)
            logger.info(
                f"Episode completed: {len(self.episode_rewards)} steps, "
                f"Total reward: {total_reward:.3f}, Mean reward: {mean_reward:.3f}"
            )
    
    def _capture_observation(self) -> np.ndarray:
        """
        Capture and preprocess current screen observation.
        
        Returns:
            Preprocessed observation as numpy array
            
        Raises:
            RuntimeError: If screen capture fails
        """
        try:
            # Capture screen
            raw_frame = grab_window(self.window_id, self.window_bounds)
            if raw_frame is None:
                raise RuntimeError("Screen capture returned None")
            
            # Preprocess for RL
            observation = prepare(raw_frame, self.input_size, normalize=True)
            
            # Ensure correct dtype and shape
            if observation.dtype != np.float32:
                observation = observation.astype(np.float32)
            
            if observation.shape != (*self.input_size, 3):
                raise RuntimeError(f"Unexpected observation shape: {observation.shape}")
            
            return observation
            
        except Exception as e:
            logger.error(f"Failed to capture observation: {e}")
            raise RuntimeError(f"Observation capture failed: {e}")
    
    def _execute_action(self, action: Action) -> bool:
        """
        Execute an action through the mouse control system.
        
        Args:
            action: Action to execute
            
        Returns:
            True if action executed successfully, False otherwise
        """
        try:
            if self.dry_run:
                logger.debug(f"DRY RUN: Would execute action {action.type}")
                return True
            
            if action.type == "noop":
                # No-op action - just wait a moment
                time.sleep(0.1)
                return True
            
            elif action.type == "click":
                # Convert grid coordinates to screen coordinates
                if self.calibration:
                    x, y = to_screen_center(action.r, action.c, self.window_bounds, self.calibration)
                else:
                    # Fallback to simple mapping if no calibration
                    x = int(self.window_bounds['X'] + (action.c / self.grid_size[1]) * self.window_bounds['Width'])
                    y = int(self.window_bounds['Y'] + (action.r / self.grid_size[0]) * self.window_bounds['Height'])
                
                click(x, y)
                return True
            
            elif action.type == "drag":
                if not action.path or len(action.path) < 2:
                    logger.warning("Drag action requires path with at least 2 points")
                    return False
                
                # Convert path coordinates to screen coordinates
                screen_path = []
                for r, c in action.path:
                    if self.calibration:
                        x, y = to_screen_center(r, c, self.window_bounds, self.calibration)
                    else:
                        x = int(self.window_bounds['X'] + (c / self.grid_size[1]) * self.window_bounds['Width'])
                        y = int(self.window_bounds['Y'] + (r / self.grid_size[0]) * self.window_bounds['Height'])
                    screen_path.append((x, y))
                
                drag_path(screen_path)
                return True
            
            elif action.type == "toolbar":
                if not self.calibration or not action.tool:
                    logger.warning("Toolbar actions require calibration and tool specification")
                    return False
                
                if action.tool in self.calibration.toolbar:
                    x, y = get_toolbar_coord(action.tool, self.window_bounds, self.calibration)
                    click(x, y)
                    return True
                else:
                    logger.warning(f"Toolbar button '{action.tool}' not calibrated")
                    return False
            
            else:
                logger.warning(f"Unknown action type: {action.type}")
                return False
                
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return False
    
    def get_reward_statistics(self) -> Dict[str, float]:
        """Get statistics about rewards in current episode."""
        base_stats = self.reward_calculator.get_reward_statistics()
        
        # Add environment-specific stats
        if self.episode_rewards:
            base_stats.update({
                'episode_total_reward': float(sum(self.episode_rewards)),
                'episode_mean_reward': float(np.mean(self.episode_rewards)),
                'episode_std_reward': float(np.std(self.episode_rewards)),
                'episode_length': len(self.episode_rewards),
            })
        
        return base_stats
    
    def get_action_meanings(self) -> list[str]:
        """Get human-readable meanings for each action."""
        action_meanings = []
        
        # Add no-op
        action_meanings.append("No-op")
        
        # Add grid clicks
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                action_meanings.append(f"Click ({r},{c})")
        
        # Add toolbar actions (if calibrated)
        if self.calibration and self.calibration.toolbar:
            for tool in self.calibration.toolbar.keys():
                action_meanings.append(f"Toolbar: {tool}")
        
        return action_meanings