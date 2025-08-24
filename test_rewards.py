#!/usr/bin/env python3
"""
Test script for the reward system.

Validates the reward calculator implementations and ensures they work correctly
before integration with the RL environment.
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from motorways.rewards import (
    ShapedRewardCalculator, 
    RewardValidationSuite,
    RewardConfig
)


def main():
    """Run reward system validation."""
    print("🧪 Testing Mini Motorways Reward System")
    print("=" * 50)
    
    # Create validation suite
    validation_suite = RewardValidationSuite(output_dir=Path("validation_results"))
    
    # Test shaped reward calculator
    print("\n📊 Testing ShapedRewardCalculator...")
    
    # Create calculator with default config
    shaped_calculator = ShapedRewardCalculator()
    
    # Run validation suite
    results = validation_suite.validate_calculator(shaped_calculator, test_episode_length=50)
    
    # Generate and print report
    report = validation_suite.generate_validation_report(results)
    print(report)
    
    # Summary
    passed_tests = sum(1 for r in results if r.passed)
    total_tests = len(results)
    
    print(f"\n🎯 Validation Summary:")
    print(f"   Tests Passed: {passed_tests}/{total_tests}")
    print(f"   Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("   ✅ All tests passed! Reward system is ready.")
    else:
        print("   ⚠️  Some tests failed. Check validation results.")
        
    print(f"\n📁 Detailed results saved to: validation_results/")
    
    # Test basic reward calculation
    print(f"\n🔍 Testing basic reward calculation...")
    shaped_calculator.reset()
    
    # Create dummy observation
    import numpy as np
    obs1 = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    obs2 = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    
    from motorways.policy.action_space import Action
    action = Action(type="click", r=10, c=15)
    
    # Calculate rewards
    result1 = shaped_calculator.calculate_reward(obs1, None, None)
    result2 = shaped_calculator.calculate_reward(obs2, obs1, action)
    
    print(f"   First step reward: {result1.total_reward:.3f}")
    print(f"   Components: {list(result1.components.keys())}")
    print(f"   Second step reward: {result2.total_reward:.3f}")
    print(f"   Components: {list(result2.components.keys())}")
    
    # Test reward statistics
    stats = shaped_calculator.get_reward_statistics()
    print(f"   Episode stats: {stats}")
    
    print("\n✅ Basic functionality test completed!")


if __name__ == "__main__":
    main()