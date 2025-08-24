"""
Base classes and interfaces for reward calculators.

Provides the foundation for all reward calculation implementations,
ensuring consistent interfaces and proper integration with the RL environment.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np
from ..policy.action_space import Action


@dataclass
class RewardResult:
    """
    Result of reward calculation including breakdown and metadata.
    
    Args:
        total_reward: The final reward value for this step
        components: Dictionary of individual reward components
        metadata: Additional information for debugging/analysis
        game_over: Whether this step indicates game over
    """
    total_reward: float
    components: Dict[str, float]
    metadata: Dict[str, Any]
    game_over: bool = False


class BaseRewardCalculator(ABC):
    """
    Base class for all reward calculators.
    
    Provides common interface and utilities that all reward implementations
    must follow. Supports both stateless and stateful reward calculation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize reward calculator.
        
        Args:
            config: Configuration dictionary for reward parameters
        """
        self.config = config or {}
        self.step_count = 0
        self._previous_observations = []
        self._reward_history = []
        
    @abstractmethod
    def calculate_reward(
        self,
        current_obs: np.ndarray,
        previous_obs: Optional[np.ndarray],
        action: Optional[Action],
        info: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """
        Calculate reward for the current step.
        
        Args:
            current_obs: Current observation (RGB image array)
            previous_obs: Previous observation (None for first step)
            action: Action taken to reach current state (None for first step)
            info: Additional context information
            
        Returns:
            RewardResult containing reward value and components
        """
        pass
    
    def reset(self):
        """Reset calculator state for new episode."""
        self.step_count = 0
        self._previous_observations.clear()
        self._reward_history.clear()
        
    def update_config(self, config: Dict[str, Any]):
        """Update configuration parameters."""
        self.config.update(config)
        
    def get_reward_statistics(self) -> Dict[str, float]:
        """Get statistics about rewards over current episode."""
        if not self._reward_history:
            return {}
            
        rewards = [r.total_reward for r in self._reward_history]
        return {
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'total_reward': float(np.sum(rewards)),
            'episode_length': len(rewards)
        }
    
    def _store_step_data(self, obs: np.ndarray, reward_result: RewardResult):
        """Store step data for analysis and temporal calculations."""
        self.step_count += 1
        
        # Keep limited history for temporal analysis
        max_history = self.config.get('max_history_frames', 10)
        if len(self._previous_observations) >= max_history:
            self._previous_observations.pop(0)
        self._previous_observations.append(obs)
        
        # Store reward history
        max_reward_history = self.config.get('max_reward_history', 1000)
        if len(self._reward_history) >= max_reward_history:
            self._reward_history.pop(0)
        self._reward_history.append(reward_result)


class CompositeRewardCalculator(BaseRewardCalculator):
    """
    Combines multiple reward calculators with configurable weights.
    
    Allows for ensemble reward systems that balance different aspects
    of the game (traffic flow, score progression, shaped rewards, etc.)
    """
    
    def __init__(self, calculators: Dict[str, BaseRewardCalculator], weights: Optional[Dict[str, float]] = None):
        """
        Initialize composite calculator.
        
        Args:
            calculators: Dictionary of reward calculators by name
            weights: Weights for combining calculators (default: equal weights)
        """
        super().__init__()
        self.calculators = calculators
        
        # Default to equal weights if not specified
        if weights is None:
            weights = {name: 1.0 / len(calculators) for name in calculators.keys()}
        self.weights = weights
        
        # Validate weights
        if set(weights.keys()) != set(calculators.keys()):
            raise ValueError("Calculator names and weight keys must match")
        
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            # Normalize weights to sum to 1.0
            self.weights = {name: w / total_weight for name, w in weights.items()}
    
    def calculate_reward(
        self,
        current_obs: np.ndarray,
        previous_obs: Optional[np.ndarray],
        action: Optional[Action],
        info: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """Calculate composite reward from all sub-calculators."""
        
        # Calculate rewards from all sub-calculators
        sub_results = {}
        all_components = {}
        all_metadata = {}
        game_over = False
        
        for name, calculator in self.calculators.items():
            try:
                result = calculator.calculate_reward(current_obs, previous_obs, action, info)
                sub_results[name] = result.total_reward
                
                # Prefix component names to avoid conflicts
                for comp_name, comp_value in result.components.items():
                    all_components[f"{name}_{comp_name}"] = comp_value
                
                # Store metadata by calculator name
                all_metadata[name] = result.metadata
                
                # Game over if any calculator detects it
                if result.game_over:
                    game_over = True
                    
            except Exception as e:
                # Log error but continue with other calculators
                sub_results[name] = 0.0
                all_metadata[name] = {'error': str(e)}
        
        # Calculate weighted combination
        total_reward = sum(
            sub_results[name] * self.weights[name]
            for name in self.calculators.keys()
        )
        
        # Add individual calculator results to components
        for name, reward in sub_results.items():
            all_components[f"calc_{name}"] = reward
        
        result = RewardResult(
            total_reward=total_reward,
            components=all_components,
            metadata={
                'calculators': all_metadata,
                'weights': self.weights,
                'sub_rewards': sub_results
            },
            game_over=game_over
        )
        
        self._store_step_data(current_obs, result)
        return result
    
    def reset(self):
        """Reset all sub-calculators."""
        super().reset()
        for calculator in self.calculators.values():
            calculator.reset()
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update calculator weights (will be normalized)."""
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            self.weights = {name: w / total_weight for name, w in new_weights.items()}
        else:
            raise ValueError("Total weight must be positive")