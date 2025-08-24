"""
Reward system for Mini Motorways RL training.

This module provides multiple reward calculation approaches:
- Shaped intermediate rewards for dense feedback
- Traffic flow efficiency rewards using computer vision
- Network connectivity and score progression rewards
- Hybrid system combining all approaches

The reward system is designed to work with the existing screen capture
and action space infrastructure.
"""

from .base import BaseRewardCalculator, RewardResult, CompositeRewardCalculator
from .shaped import ShapedRewardCalculator
from .utils import RewardConfig
from .validation import RewardValidationSuite, ValidationResult

__all__ = [
    "BaseRewardCalculator",
    "RewardResult", 
    "CompositeRewardCalculator",
    "ShapedRewardCalculator",
    "RewardConfig",
    "RewardValidationSuite",
    "ValidationResult",
]