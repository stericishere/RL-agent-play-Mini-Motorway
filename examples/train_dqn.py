#!/usr/bin/env python3
"""Example script for training a DQN model for Mini Motorways.

This script demonstrates how to:
1. Create a DQN model with recommended hyperparameters
2. Set up a training environment
3. Train the model on Mini Motorways gameplay
4. Save the trained model for inference

Usage:
    python examples/train_dqn.py --steps 100000 --save-path models/dqn_mini_motorways.zip
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

from motorways.policy.loader import create_dqn_model, get_recommended_dqn_hyperparameters
from motorways.policy.action_space import get_action_space_size, decode_action
from motorways.capture.mac_quartz import find_window, grab_window
from motorways.capture.preprocess import prepare
from motorways.control.mapping import crop_grid, to_screen_center
from motorways.control.mouse import click as mouse_click
from motorways.config.schema import Calibration

logger = logging.getLogger(__name__)


def create_training_environment(grid_h: int = 32, grid_w: int = 32, calibration_path: str = "calibration.json"):
    """Create a training environment for Mini Motorways.
    
    Note: This is a placeholder. In a real implementation, you would:
    1. Create a gymnasium environment that interfaces with the game
    2. Handle state transitions, rewards, and episode termination
    3. Implement the observation space (processed game screenshots)
    4. Implement the action space (mouse clicks, drags, toolbar selections)
    
    Args:
        grid_h: Grid height in cells
        grid_w: Grid width in cells
        
    Returns:
        Gymnasium environment compatible with Stable-Baselines3
    """
    try:
        import gymnasium as gym
        from gymnasium import spaces
        import numpy as np
        
        # Calculate action space size
        action_space_size = get_action_space_size(grid_h, grid_w)
        
        class RealMiniMotorwaysEnv(gym.Env):
            def __init__(self, calibration_path):
                super().__init__()

                # Load calibration
                self.calibration = Calibration.load(Path(calibration_path))

                # Set up window capture
                self.window_id, self.bounds = find_window("Mini Motorways")
                if self.window_id is None:
                    raise RuntimeError("Mini Motorways window not found! Make sure the game is running.")

                # Action and observation spaces
                self.action_space = spaces.Discrete(get_action_space_size(32, 32))
                self.observation_space = spaces.Box(0, 255, (3, 128, 128), dtype=np.uint8)
                
                # Episode tracking
                self.current_step = 0
                self.max_steps = 1000
                self.episode_reward = 0

            def reset(self, seed=None, options=None):
                # Reset episode state
                self.current_step = 0
                self.episode_reward = 0
                
                # TODO: Reset game to starting state
                # You might need to:
                # 1. Press a restart key/button
                # 2. Navigate to main menu and start new game
                # 3. Wait for game to fully load
                
                # For now, just capture current state
                img = grab_window(self.window_id)
                obs = crop_grid(img, self.bounds, self.calibration)
                processed = prepare(obs, (128, 128), normalize=False)
                
                info = {"step": self.current_step, "episode_reward": self.episode_reward}
                return processed.squeeze(0), info

            def step(self, action):
                # Decode and execute action
                decoded_action = decode_action(action, 32, 32)

                # Execute in game
                if decoded_action.type == "click":
                    x, y = to_screen_center(decoded_action.r, decoded_action.c, self.bounds, self.calibration)
                    mouse_click(x, y)
                elif decoded_action.type == "drag" and decoded_action.path:
                    # Handle drag actions
                    from motorways.control.mouse import drag_path
                    screen_path = []
                    for r, c in decoded_action.path:
                        x, y = to_screen_center(r, c, self.bounds, self.calibration)
                        screen_path.append((x, y))
                    drag_path(screen_path)
                elif decoded_action.type == "toolbar" and decoded_action.tool:
                    # Handle toolbar actions
                    from motorways.control.mapping import get_toolbar_coord
                    if decoded_action.tool in self.calibration.toolbar:
                        x, y = get_toolbar_coord(decoded_action.tool, self.bounds, self.calibration)
                        mouse_click(x, y)

                # Wait a moment for action to take effect
                import time
                time.sleep(0.1)

                # Capture new state
                img = grab_window(self.window_id)
                obs = crop_grid(img, self.bounds, self.calibration)
                processed = prepare(obs, (128, 128), normalize=False)

                # Calculate reward (implement your reward function)
                reward = self._calculate_reward(obs)

                # Check if episode is done
                done = self._is_episode_done(obs)
                
                self.current_step += 1

                return processed.squeeze(0), reward, done, False, {"step": self.current_step}
            
            def _calculate_reward(self, obs):
                """Calculate reward based on game state.
                
                This is a simple placeholder - you should implement proper reward based on:
                - Traffic flow efficiency
                - Cars delivered successfully  
                - Game score/milestones
                - Avoiding traffic jams
                """
                # Placeholder reward function
                base_reward = 0.1  # Small positive reward for surviving
                
                # TODO: Implement computer vision to detect:
                # - Green/red traffic signals (smooth flow vs jams)
                # - Car movement patterns
                # - Score indicators
                # - Building connections
                
                # Simple heuristic: reward based on image diversity (more activity = better)
                import cv2
                gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                activity_reward = np.sum(edges > 0) / (obs.shape[0] * obs.shape[1]) * 2.0
                
                total_reward = base_reward + activity_reward
                self.episode_reward += total_reward
                
                return total_reward
            
            def _is_episode_done(self, obs):
                """Check if episode should terminate.
                
                Episode ends when:
                - Maximum steps reached
                - Game over screen detected
                - Player manually resets game
                """
                # Check step limit
                if self.current_step >= self.max_steps:
                    logger.info(f"Episode ended: max steps ({self.max_steps}) reached")
                    return True
                
                # TODO: Implement game over detection
                # Look for game over screen, pause menu, or score screen
                # This requires analyzing the screenshot for UI elements
                
                # Placeholder: detect if image is mostly static (game paused/over)
                # You should replace this with proper game state detection
                
                return False

        
        logger.info(f"Created training environment with action space size: {action_space_size}")
        return RealMiniMotorwaysEnv(calibration_path)
        
    except ImportError as e:
        logger.error(f"Failed to create environment: {e}")
        raise ImportError("gymnasium is required for training. Install with: pip install gymnasium") from e


def train_dqn_model(
    total_timesteps: int = 100000,
    save_path: Optional[Path] = None,
    grid_h: int = 32,
    grid_w: int = 32,
    difficulty: str = "normal",
    device: str = "auto",
    calibration_path: str = "calibration.json"
) -> None:
    """Train a DQN model for Mini Motorways.
    
    Args:
        total_timesteps: Total number of training steps
        save_path: Path to save the trained model
        grid_h: Grid height in cells
        grid_w: Grid width in cells
        difficulty: Training difficulty ("easy", "normal", "hard")
        device: Device for training ("auto", "cpu", "cuda", "mps")
    """
    logger.info("Starting DQN training for Mini Motorways")
    logger.info(f"Training steps: {total_timesteps}")
    logger.info(f"Grid size: {grid_h}x{grid_w}")
    logger.info(f"Difficulty: {difficulty}")
    logger.info(f"Device: {device}")
    
    try:
        # Get recommended hyperparameters
        params = get_recommended_dqn_hyperparameters(
            grid_size=(grid_h, grid_w),
            difficulty=difficulty
        )
        
        # Create DQN model
        model = create_dqn_model(
            input_shape=params["input_shape"],
            action_space_size=params["action_space_size"],
            learning_rate=params["learning_rate"],
            buffer_size=params["buffer_size"],
            learning_starts=params["learning_starts"],
            batch_size=params["batch_size"],
            gamma=params["gamma"],
            target_update_interval=params["target_update_interval"],
            exploration_fraction=params["exploration_fraction"],
            exploration_initial_eps=params["exploration_initial_eps"],
            exploration_final_eps=params["exploration_final_eps"],
            device=device
        )
        
        # Create training environment
        env = create_training_environment(grid_h, grid_w, calibration_path)
        model.set_env(env)
        
        # Set up training callbacks (optional)
        # You could add evaluation callbacks, logging callbacks, etc.
        
        logger.info("Starting training...")
        model.learn(total_timesteps=total_timesteps, log_interval=10)
        logger.info("Training completed!")
        
        # Save the model
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(save_path)
            logger.info(f"Model saved to: {save_path}")
        
        # Evaluate the trained model (optional)
        logger.info("Evaluating trained model...")
        evaluate_model(model, env, episodes=5)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def evaluate_model(model, env, episodes: int = 5) -> None:
    """Evaluate a trained model.
    
    Args:
        model: Trained DQN model
        env: Evaluation environment
        episodes: Number of episodes to run
    """
    logger.info(f"Running evaluation for {episodes} episodes")
    
    total_reward = 0
    episode_rewards = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Get action from trained model (deterministic)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        total_reward += episode_reward
        logger.info(f"Episode {episode + 1}: reward = {episode_reward:.2f}")
    
    avg_reward = total_reward / episodes
    logger.info(f"Evaluation complete. Average reward: {avg_reward:.2f}")
    
    return avg_reward


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train DQN model for Mini Motorways")
    
    parser.add_argument(
        "--steps", type=int, default=100000,
        help="Total training timesteps (default: 100000)"
    )
    parser.add_argument(
        "--save-path", type=Path, default="models/dqn_mini_motorways.zip",
        help="Path to save trained model (default: models/dqn_mini_motorways.zip)"
    )
    parser.add_argument(
        "--grid-h", type=int, default=32,
        help="Grid height in cells (default: 32)"
    )
    parser.add_argument(
        "--grid-w", type=int, default=32,
        help="Grid width in cells (default: 32)"
    )
    parser.add_argument(
        "--difficulty", choices=["easy", "normal", "hard"], default="normal",
        help="Training difficulty (default: normal)"
    )
    parser.add_argument(
        "--device", default="auto",
        help="Training device (default: auto)"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--calibration", type=str, default="calibration.json",
        help="Path to calibration file (default: calibration.json)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Train the model
    train_dqn_model(
        total_timesteps=args.steps,
        save_path=args.save_path,
        grid_h=args.grid_h,
        grid_w=args.grid_w,
        difficulty=args.difficulty,
        device=args.device,
        calibration_path=args.calibration
    )


if __name__ == "__main__":
    main()