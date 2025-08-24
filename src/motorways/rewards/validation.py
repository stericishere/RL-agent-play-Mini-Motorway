"""
Validation and testing framework for reward systems.

Provides tools to validate reward calculators, test their behavior,
and ensure they work correctly with the RL environment.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import json
import time

from .base import BaseRewardCalculator, RewardResult
from .shaped import ShapedRewardCalculator
from ..policy.action_space import Action, get_action_space_size


@dataclass
class ValidationResult:
    """Result of reward calculator validation."""
    
    calculator_name: str
    test_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    execution_time: float


class RewardValidationSuite:
    """
    Comprehensive validation suite for reward calculators.
    
    Tests reward calculators for correctness, consistency, and performance
    to ensure they work properly in RL training environments.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize validation suite.
        
        Args:
            output_dir: Directory to save validation results and plots
        """
        self.output_dir = output_dir or Path("validation_results")
        self.output_dir.mkdir(exist_ok=True)
        
    def validate_calculator(self, calculator: BaseRewardCalculator, 
                          test_episode_length: int = 100) -> List[ValidationResult]:
        """
        Run full validation suite on a reward calculator.
        
        Args:
            calculator: The reward calculator to validate
            test_episode_length: Length of test episodes to simulate
            
        Returns:
            List of validation results for different tests
        """
        results = []
        calc_name = calculator.__class__.__name__
        
        # Test 1: Basic functionality
        results.append(self._test_basic_functionality(calculator))
        
        # Test 2: Consistency across resets
        results.append(self._test_reset_consistency(calculator))
        
        # Test 3: Reward stability
        results.append(self._test_reward_stability(calculator, test_episode_length))
        
        # Test 4: Performance benchmarking
        results.append(self._test_performance(calculator))
        
        # Test 5: Component balance
        results.append(self._test_component_balance(calculator, test_episode_length))
        
        # Test 6: Game over detection
        results.append(self._test_game_over_detection(calculator))
        
        # Save results
        self._save_validation_results(calc_name, results)
        
        return results
    
    def _test_basic_functionality(self, calculator: BaseRewardCalculator) -> ValidationResult:
        """Test basic reward calculation functionality."""
        start_time = time.time()
        
        try:
            # Create dummy observations
            obs1 = self._create_dummy_observation()
            obs2 = self._create_dummy_observation(variation=0.1)
            action = Action(type="click", r=10, c=15)
            
            # Test first step (no previous observation)
            result1 = calculator.calculate_reward(obs1, None, None)
            
            # Test normal step
            result2 = calculator.calculate_reward(obs2, obs1, action)
            
            # Basic validations
            checks = {
                'result1_is_RewardResult': isinstance(result1, RewardResult),
                'result2_is_RewardResult': isinstance(result2, RewardResult),
                'reward1_is_number': isinstance(result1.total_reward, (int, float)),
                'reward2_is_number': isinstance(result2.total_reward, (int, float)),
                'components1_is_dict': isinstance(result1.components, dict),
                'components2_is_dict': isinstance(result2.components, dict),
                'metadata1_is_dict': isinstance(result1.metadata, dict),
                'metadata2_is_dict': isinstance(result2.metadata, dict),
                'reward1_finite': np.isfinite(result1.total_reward),
                'reward2_finite': np.isfinite(result2.total_reward),
            }
            
            passed = all(checks.values())
            score = sum(checks.values()) / len(checks)
            
            execution_time = time.time() - start_time
            
            return ValidationResult(
                calculator_name=calculator.__class__.__name__,
                test_name="basic_functionality",
                passed=passed,
                score=score,
                details={
                    'checks': checks,
                    'reward1': result1.total_reward,
                    'reward2': result2.total_reward,
                    'components1': list(result1.components.keys()),
                    'components2': list(result2.components.keys()),
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ValidationResult(
                calculator_name=calculator.__class__.__name__,
                test_name="basic_functionality",
                passed=False,
                score=0.0,
                details={'error': str(e)},
                execution_time=execution_time
            )
    
    def _test_reset_consistency(self, calculator: BaseRewardCalculator) -> ValidationResult:
        """Test that calculator behaves consistently after resets."""
        start_time = time.time()
        
        try:
            obs = self._create_dummy_observation()
            action = Action(type="click", r=5, c=8)
            
            # Calculate reward, then reset and calculate again
            result1 = calculator.calculate_reward(obs, None, None)
            calculator.reset()
            result2 = calculator.calculate_reward(obs, None, None)
            
            # Check consistency (should be very similar, allowing for small numerical differences)
            reward_diff = abs(result1.total_reward - result2.total_reward)
            consistency_threshold = 1e-6
            
            checks = {
                'rewards_consistent': reward_diff < consistency_threshold,
                'components_same_keys': set(result1.components.keys()) == set(result2.components.keys()),
                'step_count_reset': calculator.step_count == 1,  # Should be 1 after reset + 1 calculation
            }
            
            passed = all(checks.values())
            score = sum(checks.values()) / len(checks)
            
            execution_time = time.time() - start_time
            
            return ValidationResult(
                calculator_name=calculator.__class__.__name__,
                test_name="reset_consistency",
                passed=passed,
                score=score,
                details={
                    'checks': checks,
                    'reward_difference': reward_diff,
                    'result1_reward': result1.total_reward,
                    'result2_reward': result2.total_reward,
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ValidationResult(
                calculator_name=calculator.__class__.__name__,
                test_name="reset_consistency",
                passed=False,
                score=0.0,
                details={'error': str(e)},
                execution_time=execution_time
            )
    
    def _test_reward_stability(self, calculator: BaseRewardCalculator, 
                             episode_length: int) -> ValidationResult:
        """Test reward stability over a simulated episode."""
        start_time = time.time()
        
        try:
            calculator.reset()
            rewards = []
            previous_obs = None
            
            # Simulate an episode
            for step in range(episode_length):
                obs = self._create_dummy_observation(variation=step * 0.01)
                action = self._create_dummy_action(step)
                
                result = calculator.calculate_reward(obs, previous_obs, action)
                rewards.append(result.total_reward)
                previous_obs = obs
            
            # Analyze reward stability
            rewards_array = np.array(rewards)
            
            checks = {
                'all_finite': np.all(np.isfinite(rewards_array)),
                'no_extreme_values': np.all(np.abs(rewards_array) < 1000),  # Reasonable range
                'reasonable_variance': np.std(rewards_array) < 100,  # Not too volatile
                'non_constant': np.std(rewards_array) > 1e-6,  # Should have some variation
            }
            
            passed = all(checks.values())
            score = sum(checks.values()) / len(checks)
            
            execution_time = time.time() - start_time
            
            return ValidationResult(
                calculator_name=calculator.__class__.__name__,
                test_name="reward_stability",
                passed=passed,
                score=score,
                details={
                    'checks': checks,
                    'reward_stats': {
                        'mean': float(np.mean(rewards_array)),
                        'std': float(np.std(rewards_array)),
                        'min': float(np.min(rewards_array)),
                        'max': float(np.max(rewards_array)),
                    },
                    'episode_length': episode_length,
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ValidationResult(
                calculator_name=calculator.__class__.__name__,
                test_name="reward_stability",
                passed=False,
                score=0.0,
                details={'error': str(e)},
                execution_time=execution_time
            )
    
    def _test_performance(self, calculator: BaseRewardCalculator) -> ValidationResult:
        """Test performance of reward calculation."""
        start_time = time.time()
        
        try:
            obs1 = self._create_dummy_observation()
            obs2 = self._create_dummy_observation(variation=0.1)
            action = Action(type="drag", path=[(5, 5), (10, 10)])
            
            # Measure time for multiple calculations
            num_iterations = 100
            calc_start = time.time()
            
            for _ in range(num_iterations):
                calculator.calculate_reward(obs2, obs1, action)
            
            calc_time = time.time() - calc_start
            avg_time_per_calculation = calc_time / num_iterations
            
            # Performance targets
            target_time_per_calc = 0.01  # 10ms per calculation (100 Hz)
            
            checks = {
                'meets_performance_target': avg_time_per_calculation < target_time_per_calc,
                'reasonable_speed': avg_time_per_calculation < 0.1,  # At least 10 Hz
            }
            
            passed = all(checks.values())
            score = min(1.0, target_time_per_calc / max(avg_time_per_calculation, 1e-10)) if avg_time_per_calculation is not None else 1.0
            
            execution_time = time.time() - start_time
            
            return ValidationResult(
                calculator_name=calculator.__class__.__name__,
                test_name="performance",
                passed=passed,
                score=score,
                details={
                    'checks': checks,
                    'avg_time_per_calculation': avg_time_per_calculation,
                    'target_time': target_time_per_calc,
                    'calculations_per_second': 1.0 / max(avg_time_per_calculation, 1e-10) if avg_time_per_calculation is not None else float('inf'),
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ValidationResult(
                calculator_name=calculator.__class__.__name__,
                test_name="performance",
                passed=False,
                score=0.0,
                details={'error': str(e)},
                execution_time=execution_time
            )
    
    def _test_component_balance(self, calculator: BaseRewardCalculator, 
                              episode_length: int) -> ValidationResult:
        """Test that reward components are balanced and meaningful."""
        start_time = time.time()
        
        try:
            calculator.reset()
            all_components = {}
            previous_obs = None
            
            # Collect component data over episode
            for step in range(episode_length):
                obs = self._create_dummy_observation(variation=step * 0.01)
                action = self._create_dummy_action(step)
                
                result = calculator.calculate_reward(obs, previous_obs, action)
                
                for comp_name, comp_value in result.components.items():
                    if comp_name not in all_components:
                        all_components[comp_name] = []
                    all_components[comp_name].append(comp_value)
                
                previous_obs = obs
            
            # Analyze component balance
            component_stats = {}
            for comp_name, values in all_components.items():
                values_array = np.array(values)
                component_stats[comp_name] = {
                    'mean': float(np.mean(values_array)),
                    'std': float(np.std(values_array)),
                    'contribution': float(np.mean(np.abs(values_array))),
                }
            
            # Check for balance (no single component dominates)
            contributions = [stats['contribution'] for stats in component_stats.values()]
            max_contrib = max(contributions) if contributions else 0.0
            total_contrib = max(sum(contributions), 1e-10) if contributions else 1.0
            
            checks = {
                'has_components': len(all_components) > 0,
                'multiple_components': len(all_components) > 1,
                'no_dominating_component': max_contrib / total_contrib < 0.8 if total_contrib > 0 else True,
                'components_contribute': all(c > 1e-6 for c in contributions),
            }
            
            passed = all(checks.values())
            score = sum(checks.values()) / len(checks)
            
            execution_time = time.time() - start_time
            
            return ValidationResult(
                calculator_name=calculator.__class__.__name__,
                test_name="component_balance",
                passed=passed,
                score=score,
                details={
                    'checks': checks,
                    'component_stats': component_stats,
                    'max_contribution_ratio': max_contrib / total_contrib,
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ValidationResult(
                calculator_name=calculator.__class__.__name__,
                test_name="component_balance",
                passed=False,
                score=0.0,
                details={'error': str(e)},
                execution_time=execution_time
            )
    
    def _test_game_over_detection(self, calculator: BaseRewardCalculator) -> ValidationResult:
        """Test game over detection capability."""
        start_time = time.time()
        
        try:
            calculator.reset()
            
            # Test normal conditions (should not trigger game over)
            normal_obs = self._create_dummy_observation()
            normal_result = calculator.calculate_reward(normal_obs, None, None)
            
            # Simulate potentially problematic conditions
            game_over_detected = False
            for step in range(50):
                # Create low-activity observation
                low_activity_obs = self._create_dummy_observation(variation=0.001)
                no_action = Action(type="noop")
                
                result = calculator.calculate_reward(low_activity_obs, normal_obs, no_action)
                if result.game_over:
                    game_over_detected = True
                    break
            
            checks = {
                'normal_no_game_over': not normal_result.game_over,
                'has_game_over_detection': hasattr(calculator, '_detect_game_over'),
                'game_over_field_present': hasattr(normal_result, 'game_over'),
            }
            
            passed = all(checks.values())
            score = sum(checks.values()) / len(checks)
            
            execution_time = time.time() - start_time
            
            return ValidationResult(
                calculator_name=calculator.__class__.__name__,
                test_name="game_over_detection",
                passed=passed,
                score=score,
                details={
                    'checks': checks,
                    'game_over_detected_in_test': game_over_detected,
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ValidationResult(
                calculator_name=calculator.__class__.__name__,
                test_name="game_over_detection",
                passed=False,
                score=0.0,
                details={'error': str(e)},
                execution_time=execution_time
            )
    
    def _create_dummy_observation(self, size: Tuple[int, int] = (128, 128), variation: float = 0.0) -> np.ndarray:
        """Create dummy observation for testing."""
        obs = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
        
        # Add some structure to make it more realistic
        # Add some "roads" (dark lines)
        obs[size[0]//4:size[0]//4+2, :] = [64, 64, 64]  # Horizontal road
        obs[:, size[1]//4:size[1]//4+2] = [64, 64, 64]  # Vertical road
        
        # Add some "buildings" (colored squares)
        obs[10:20, 10:20] = [255, 0, 0]  # Red building
        obs[30:40, 30:40] = [0, 255, 0]  # Green building
        obs[50:60, 50:60] = [0, 0, 255]  # Blue building
        
        # Add variation if requested
        if variation > 0:
            noise = np.random.normal(0, variation * 255, obs.shape).astype(np.int16)
            obs = np.clip(obs.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return obs
    
    def _create_dummy_action(self, step: int) -> Action:
        """Create dummy action for testing."""
        action_types = ["click", "drag", "noop", "toolbar"]
        action_type = action_types[step % len(action_types)]
        
        if action_type == "click":
            return Action(type="click", r=step % 32, c=(step * 2) % 32)
        elif action_type == "drag":
            start_r, start_c = step % 32, (step * 2) % 32
            end_r, end_c = (step + 5) % 32, (step * 2 + 5) % 32
            return Action(type="drag", path=[(start_r, start_c), (end_r, end_c)])
        elif action_type == "toolbar":
            tools = ["road", "bridge", "roundabout", "traffic_light"]
            return Action(type="toolbar", tool=tools[step % len(tools)])
        else:
            return Action(type="noop")
    
    def _save_validation_results(self, calculator_name: str, results: List[ValidationResult]):
        """Save validation results to file."""
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        results_dict = {
            'calculator_name': calculator_name,
            'timestamp': time.time(),
            'summary': {
                'total_tests': len(results),
                'passed_tests': sum(1 for r in results if r.passed),
                'average_score': sum(r.score for r in results) / len(results) if results else 0.0,
                'total_execution_time': sum(r.execution_time for r in results),
            },
            'tests': [
                {
                    'test_name': r.test_name,
                    'passed': bool(r.passed),
                    'score': float(r.score),
                    'execution_time': float(r.execution_time),
                    'details': convert_numpy_types(r.details),
                }
                for r in results
            ]
        }
        
        output_file = self.output_dir / f"{calculator_name}_validation.json"
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
    
    def generate_validation_report(self, results: List[ValidationResult]) -> str:
        """Generate human-readable validation report."""
        report = []
        report.append("=" * 60)
        report.append("REWARD CALCULATOR VALIDATION REPORT")
        report.append("=" * 60)
        
        if not results:
            report.append("No validation results provided.")
            return "\n".join(report)
        
        calculator_name = results[0].calculator_name
        report.append(f"Calculator: {calculator_name}")
        report.append("")
        
        # Summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        avg_score = sum(r.score for r in results) / total_tests
        total_time = sum(r.execution_time for r in results)
        
        report.append(f"Summary:")
        report.append(f"  Tests Run: {total_tests}")
        report.append(f"  Tests Passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
        report.append(f"  Average Score: {avg_score:.3f}")
        report.append(f"  Total Execution Time: {total_time:.3f}s")
        report.append("")
        
        # Individual test results
        report.append("Individual Test Results:")
        report.append("-" * 40)
        
        for result in results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            report.append(f"{result.test_name}: {status} (Score: {result.score:.3f}, Time: {result.execution_time:.3f}s)")
            
            if not result.passed and 'error' in result.details:
                report.append(f"  Error: {result.details['error']}")
            elif 'checks' in result.details:
                failed_checks = [k for k, v in result.details['checks'].items() if not v]
                if failed_checks:
                    report.append(f"  Failed checks: {', '.join(failed_checks)}")
        
        return "\n".join(report)