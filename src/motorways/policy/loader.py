"""Policy loading and inference for SB3 and PyTorch models."""

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from motorways.config.schema import Action
from motorways.policy.action_space import decode_action

logger = logging.getLogger(__name__)


def load_model(
    model_path: Path,
    device: str = "cpu",
    grid_h: int = 32,
    grid_w: int = 32
) -> Callable[[np.ndarray], Action]:
    """Load trained model and return prediction function.

    Args:
        model_path: Path to model file (.zip for SB3, .pt/.pth for PyTorch)
        device: Device to run inference on ("cpu", "mps", "cuda")
        grid_h: Grid height for action decoding
        grid_w: Grid width for action decoding

    Returns:
        Function that takes observation and returns Action

    Raises:
        ImportError: If required dependencies are missing
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_path = Path(model_path)

    # Try SB3 loading first for .zip files
    if model_path.suffix == ".zip":
        try:
            from stable_baselines3 import PPO, DQN
            from stable_baselines3.common.policies import BasePolicy

            logger.info(f"Loading SB3 model from {model_path}")
            
            # Try to load as DQN first, then fallback to PPO
            model = None
            model_type = None
            
            try:
                model = DQN.load(model_path, device=device)
                model_type = "DQN"
                logger.info("Successfully loaded as DQN model")
            except Exception as dqn_error:
                logger.debug(f"Failed to load as DQN: {dqn_error}")
                try:
                    model = PPO.load(model_path, device=device)
                    model_type = "PPO"
                    logger.info("Successfully loaded as PPO model")
                except Exception as ppo_error:
                    logger.error(f"Failed to load as PPO: {ppo_error}")
                    raise RuntimeError(
                        f"Failed to load model as either DQN or PPO. "
                        f"DQN error: {dqn_error}, PPO error: {ppo_error}"
                    ) from ppo_error

            def predict_sb3(obs: np.ndarray) -> Action:
                """SB3 model prediction function (supports both DQN and PPO)."""
                try:
                    # Ensure observation has correct shape and type
                    if obs.dtype != np.float32:
                        obs = obs.astype(np.float32)

                    # SB3 expects batch dimension
                    if obs.ndim == 3:  # (C, H, W)
                        obs = np.expand_dims(obs, axis=0)  # (1, C, H, W)

                    # DQN and PPO both support predict() with deterministic flag
                    if model_type == "DQN":
                        # DQN uses deterministic=True to disable epsilon-greedy exploration
                        action_value, _ = model.predict(obs, deterministic=True)
                        logger.debug(f"DQN prediction (deterministic=True)")
                    else:  # PPO
                        action_value, _ = model.predict(obs, deterministic=True)
                        logger.debug(f"PPO prediction (deterministic=True)")

                    # Handle both scalar and array outputs
                    if isinstance(action_value, np.ndarray):
                        action_value = int(action_value.item())
                    else:
                        action_value = int(action_value)

                    decoded_action = decode_action(action_value, grid_h, grid_w)
                    logger.debug(
                        f"{model_type} predicted action {action_value} -> "
                        f"{decoded_action.type}"
                    )
                    return decoded_action

                except Exception as e:
                    logger.error(f"SB3 {model_type} prediction failed: {e}")
                    return Action(type="noop")

            return predict_sb3

        except ImportError as e:
            logger.error(f"SB3 not available: {e}")
            raise ImportError("stable-baselines3 is required for .zip model files") from e
        except Exception as e:
            logger.error(f"Failed to load SB3 model: {e}")
            raise RuntimeError(f"SB3 model loading failed: {e}") from e

    # Try PyTorch loading for .pt/.pth files
    elif model_path.suffix in [".pt", ".pth"]:
        try:
            logger.info(f"Loading PyTorch model from {model_path}")

            # Load model
            if device == "mps" and torch.backends.mps.is_available():
                map_location = torch.device("mps")
            elif device == "cuda" and torch.cuda.is_available():
                map_location = torch.device("cuda")
            else:
                map_location = torch.device("cpu")

            model = torch.load(model_path, map_location=map_location)

            # Handle different model formats
            if hasattr(model, 'eval'):
                model.eval()
                net = model
            elif isinstance(model, dict) and 'model' in model:
                net = model['model']
                net.eval()
            elif isinstance(model, dict) and 'state_dict' in model:
                # Need to reconstruct model from state dict - this is tricky without knowing architecture
                logger.error("State dict loading not implemented - please provide full model")
                raise RuntimeError("State dict loading requires model architecture")
            else:
                net = model
                if hasattr(net, 'eval'):
                    net.eval()

            def predict_torch(obs: np.ndarray) -> Action:
                """PyTorch model prediction function."""
                try:
                    # Convert to torch tensor
                    if obs.dtype != np.float32:
                        obs = obs.astype(np.float32)

                    # Ensure batch dimension
                    if obs.ndim == 3:  # (C, H, W)
                        obs = np.expand_dims(obs, axis=0)  # (1, C, H, W)

                    obs_tensor = torch.from_numpy(obs).to(map_location)

                    with torch.no_grad():
                        outputs = net(obs_tensor)

                        # Handle different output formats
                        if isinstance(outputs, tuple):
                            logits = outputs[0]
                        else:
                            logits = outputs

                        # Get action with highest probability
                        action_value = torch.argmax(logits, dim=-1).item()

                    decoded_action = decode_action(action_value, grid_h, grid_w)
                    logger.debug(f"PyTorch predicted action {action_value} -> {decoded_action.type}")
                    return decoded_action

                except Exception as e:
                    logger.error(f"PyTorch prediction failed: {e}")
                    return Action(type="noop")

            return predict_torch

        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            raise RuntimeError(f"PyTorch model loading failed: {e}") from e

    else:
        raise ValueError(f"Unsupported model file extension: {model_path.suffix}. Supported: .zip (SB3), .pt/.pth (PyTorch)")


def create_random_policy(grid_h: int = 32, grid_w: int = 32) -> Callable[[np.ndarray], Action]:
    """Create a random policy for testing.

    Args:
        grid_h: Grid height for action space
        grid_w: Grid width for action space

    Returns:
        Function that returns random actions
    """
    from motorways.policy.action_space import sample_random_action

    def random_predict(obs: np.ndarray) -> Action:
        """Random policy prediction."""
        return sample_random_action(grid_h, grid_w)

    logger.info(f"Created random policy for {grid_h}x{grid_w} grid")
    return random_predict


def validate_model_compatibility(
    model_path: Path,
    expected_input_shape: tuple,
    expected_action_space_size: int
) -> bool:
    """Validate model compatibility with expected input/output shapes.

    Args:
        model_path: Path to model file
        expected_input_shape: Expected input shape (C, H, W)
        expected_action_space_size: Expected number of actions

    Returns:
        True if model appears compatible
    """
    try:
        if model_path.suffix == ".zip":
            # For SB3 models, we can check observation space
            try:
                from stable_baselines3 import PPO, DQN
                
                # Try DQN first, then PPO
                model = None
                try:
                    model = DQN.load(model_path, device="cpu")
                    logger.debug("Validation: loaded as DQN")
                except Exception:
                    model = PPO.load(model_path, device="cpu")
                    logger.debug("Validation: loaded as PPO")

                obs_space = model.observation_space
                if hasattr(obs_space, 'shape'):
                    actual_shape = obs_space.shape
                    if actual_shape != expected_input_shape:
                        logger.warning(f"Model input shape {actual_shape} != expected {expected_input_shape}")
                        return False

                action_space = model.action_space
                if hasattr(action_space, 'n'):
                    actual_action_size = action_space.n
                    if actual_action_size != expected_action_space_size:
                        logger.warning(f"Model action space {actual_action_size} != expected {expected_action_space_size}")
                        return False

                logger.info("SB3 model compatibility validated")
                return True

            except Exception as e:
                logger.error(f"SB3 compatibility check failed: {e}")
                return False

        elif model_path.suffix in [".pt", ".pth"]:
            # For PyTorch models, validation is more limited
            try:
                model = torch.load(model_path, map_location="cpu")

                # Basic check that we can load the model
                if hasattr(model, 'eval'):
                    logger.info("PyTorch model appears loadable")
                    return True
                elif isinstance(model, dict):
                    logger.info("PyTorch model dict format detected")
                    return True
                else:
                    logger.warning("PyTorch model format unclear")
                    return False

            except Exception as e:
                logger.error(f"PyTorch compatibility check failed: {e}")
                return False

        else:
            logger.error(f"Unknown model format: {model_path.suffix}")
            return False

    except Exception as e:
        logger.error(f"Model compatibility validation failed: {e}")
        return False


def create_dqn_model(
    input_shape: tuple[int, int, int],
    action_space_size: int,
    learning_rate: float = 1e-4,
    buffer_size: int = 50000,
    learning_starts: int = 1000,
    batch_size: int = 32,
    gamma: float = 0.99,
    target_update_interval: int = 1000,
    exploration_fraction: float = 0.1,
    exploration_initial_eps: float = 1.0,
    exploration_final_eps: float = 0.05,
    device: str = "auto"
):
    """Create a DQN model for Mini Motorways using Stable-Baselines3.
    
    Args:
        input_shape: Input observation shape (C, H, W)
        action_space_size: Number of discrete actions
        learning_rate: Learning rate for optimizer
        buffer_size: Replay buffer size
        learning_starts: Steps before learning starts
        batch_size: Batch size for training
        gamma: Discount factor
        target_update_interval: Steps between target network updates
        exploration_fraction: Fraction of training for exploration decay
        exploration_initial_eps: Initial epsilon for exploration
        exploration_final_eps: Final epsilon for exploration
        device: Device for training ("auto", "cpu", "cuda", "mps")
        
    Returns:
        Configured DQN model ready for training
        
    Raises:
        ImportError: If stable-baselines3 is not available
    """
    try:
        from stable_baselines3 import DQN
        from gymnasium import spaces
        
        logger.info(f"Creating DQN model for Mini Motorways")
        logger.info(f"Input shape: {input_shape}, Action space: {action_space_size}")
        
        # Create observation and action spaces
        observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=input_shape,
            dtype=np.float32
        )
        action_space = spaces.Discrete(action_space_size)
        
        # Create DQN model with optimized hyperparameters for Mini Motorways
        model = DQN(
            policy="CnnPolicy",  # Convolutional neural network for image input
            env=None,  # Will be set during training
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            device=device,
            verbose=1,
            policy_kwargs={
                "features_extractor_class": "CnnFeaturesExtractor",
                "net_arch": [512, 512],  # Dense layers after CNN
                "activation_fn": torch.nn.ReLU,
            }
        )
        
        logger.info("DQN model created successfully")
        logger.info(f"Policy: {model.policy}")
        logger.info(f"Device: {model.device}")
        
        return model
        
    except ImportError as e:
        logger.error(f"Failed to import stable-baselines3: {e}")
        raise ImportError(
            "stable-baselines3 is required for DQN model creation. "
            "Install with: pip install stable-baselines3[extra]"
        ) from e
    except Exception as e:
        logger.error(f"Failed to create DQN model: {e}")
        raise RuntimeError(f"DQN model creation failed: {e}") from e


def get_recommended_dqn_hyperparameters(
    grid_size: tuple[int, int] = (32, 32),
    difficulty: str = "normal"
) -> dict:
    """Get recommended DQN hyperparameters for Mini Motorways.
    
    Args:
        grid_size: Grid dimensions (height, width)
        difficulty: Difficulty level ("easy", "normal", "hard")
        
    Returns:
        Dictionary of recommended hyperparameters
    """
    from motorways.policy.action_space import get_action_space_size
    
    grid_h, grid_w = grid_size
    action_space_size = get_action_space_size(grid_h, grid_w)
    
    base_params = {
        "learning_rate": 1e-4,
        "buffer_size": 50000,
        "learning_starts": 5000,
        "batch_size": 32,
        "gamma": 0.99,
        "target_update_interval": 1000,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    }
    
    # Adjust parameters based on difficulty
    if difficulty == "easy":
        base_params.update({
            "learning_rate": 2e-4,
            "exploration_final_eps": 0.1,
            "gamma": 0.95,
        })
    elif difficulty == "hard":
        base_params.update({
            "learning_rate": 5e-5,
            "buffer_size": 100000,
            "learning_starts": 10000,
            "exploration_final_eps": 0.02,
            "gamma": 0.995,
        })
    
    base_params["action_space_size"] = action_space_size
    base_params["input_shape"] = (3, 128, 128)  # RGB image
    
    logger.info(f"Recommended DQN hyperparameters for {difficulty} difficulty:")
    for key, value in base_params.items():
        logger.info(f"  {key}: {value}")
    
    return base_params


def get_device_recommendation() -> str:
    """Get recommended device for inference.

    Returns:
        Recommended device string ("cpu", "mps", "cuda")
    """
    try:
        import torch

        # Check for Apple Silicon MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Apple MPS available - recommended for Apple Silicon")
            return "mps"

        # Check for CUDA
        elif torch.cuda.is_available():
            logger.info("CUDA available - using GPU acceleration")
            return "cuda"

        else:
            logger.info("Using CPU inference")
            return "cpu"

    except ImportError:
        logger.info("PyTorch not available - defaulting to CPU")
        return "cpu"
