#!/usr/bin/env python3
"""
Test script for the Mini Motorways Gymnasium environment.

Tests the environment functionality including observation capture,
action execution, reward calculation, and episode management.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from motorways.environment import MiniMotorwaysEnv
from motorways.rewards import RewardConfig


def test_environment_creation():
    """Test basic environment creation and configuration."""
    print("🏗️ Testing environment creation...")
    
    # Test with default configuration
    env = MiniMotorwaysEnv(dry_run=True)  # Dry run to avoid mouse clicks
    
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Max episode steps: {env.max_episode_steps}")
    
    # Test with custom reward configuration
    reward_config = {
        'survival_reward': 0.2,
        'construction_reward_scale': 2.0,
    }
    
    env_custom = MiniMotorwaysEnv(
        reward_config=reward_config,
        max_episode_steps=500,
        dry_run=True
    )
    
    print(f"   Custom env action space: {env_custom.action_space}")
    print("   ✅ Environment creation successful")
    
    return env


def test_environment_interface():
    """Test the Gymnasium interface compliance."""
    print("\n🔧 Testing Gymnasium interface compliance...")
    
    env = MiniMotorwaysEnv(dry_run=True)
    
    # Test action space
    assert hasattr(env, 'action_space'), "Missing action_space"
    assert hasattr(env, 'observation_space'), "Missing observation_space"
    
    # Test required methods
    required_methods = ['reset', 'step', 'render', 'close']
    for method in required_methods:
        assert hasattr(env, method), f"Missing method: {method}"
    
    # Test action space properties
    assert env.action_space.n > 0, "Action space should have positive size"
    
    # Test observation space properties
    obs_shape = env.observation_space.shape
    assert len(obs_shape) == 3, "Observation should be 3D (H, W, C)"
    assert obs_shape[2] == 3, "Observation should have 3 channels (RGB)"
    
    print("   ✅ Interface compliance test passed")


def test_environment_without_window():
    """Test environment behavior when game window is not available."""
    print("\n🪟 Testing environment without game window...")
    
    env = MiniMotorwaysEnv(window_title="NonExistentWindow", dry_run=True)
    
    try:
        obs, info = env.reset()
        print("   ⚠️  Environment reset unexpectedly succeeded without window")
    except RuntimeError as e:
        print(f"   ✅ Properly caught window error: {str(e)[:60]}...")
    
    env.close()


def test_action_meanings():
    """Test action meaning generation."""
    print("\n📝 Testing action meanings...")
    
    env = MiniMotorwaysEnv(dry_run=True)
    meanings = env.get_action_meanings()
    
    print(f"   Total actions: {len(meanings)}")
    print(f"   First 5 actions: {meanings[:5]}")
    print(f"   Action space size: {env.action_space.n}")
    
    # Should have at least no-op + grid actions
    expected_min = 1 + (32 * 32)  # no-op + 32x32 grid
    assert len(meanings) >= expected_min, f"Expected at least {expected_min} actions"
    
    print("   ✅ Action meanings test passed")


def test_reward_statistics():
    """Test reward statistics functionality."""
    print("\n📊 Testing reward statistics...")
    
    env = MiniMotorwaysEnv(dry_run=True)
    
    # Test with no rewards yet
    stats = env.get_reward_statistics()
    print(f"   Initial stats: {stats}")
    
    # Simulate some episode rewards
    env.episode_rewards = [1.0, -0.5, 2.0, 0.0, -1.0]
    stats = env.get_reward_statistics()
    
    print(f"   Episode stats: {stats}")
    
    expected_keys = ['episode_total_reward', 'episode_mean_reward', 'episode_length']
    for key in expected_keys:
        assert key in stats, f"Missing stat: {key}"
    
    assert stats['episode_length'] == 5, "Incorrect episode length"
    assert abs(stats['episode_total_reward'] - 1.5) < 1e-6, "Incorrect total reward"
    
    print("   ✅ Reward statistics test passed")


def test_dry_run_functionality():
    """Test that dry run mode works correctly."""
    print("\n🧪 Testing dry run functionality...")
    
    env = MiniMotorwaysEnv(dry_run=True)
    
    # Create mock observation data
    obs = np.random.random((128, 128, 3)).astype(np.float32)
    
    # Test action execution in dry run
    from motorways.policy.action_space import Action
    
    actions = [
        Action(type="noop"),
        Action(type="click", r=10, c=15),
        Action(type="drag", path=[(5, 5), (10, 10)]),
    ]
    
    for action in actions:
        success = env._execute_action(action)
        print(f"   Action {action.type}: {'✅' if success else '❌'}")
        assert success, f"Action {action.type} should succeed in dry run"
    
    print("   ✅ Dry run functionality test passed")


def main():
    """Run all environment tests."""
    print("🧪 Testing Mini Motorways Gymnasium Environment")
    print("=" * 60)
    
    try:
        # Run tests
        env = test_environment_creation()
        test_environment_interface()
        test_environment_without_window()
        test_action_meanings()
        test_reward_statistics()
        test_dry_run_functionality()
        
        # Cleanup
        env.close()
        
        print("\n🎉 All environment tests passed!")
        print("✅ The Gymnasium environment is ready for RL training")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()