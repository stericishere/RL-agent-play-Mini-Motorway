# Mini Motorways RL Player

[![GitHub](https://img.shields.io/badge/GitHub-MotorwaysRL-blue?logo=github)]([https://github.com/your-username/motorways-rl](https://github.com/stericishere/RL-agent-play-Mini-Motorways))
<img width="1436" height="895" alt="Screenshot 2025-08-24 at 19 13 34" src="https://github.com/user-attachments/assets/7df1f5b0-d61f-42b9-8568-03890fd50e8e" />

<!-- Tech Stack Badges -->
<p align="center">
  <!-- Core -->
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/macOS-000000?logo=apple&logoColor=white" alt="macOS"/>

  <!-- ML/RL -->
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Stable%20Baselines3-43B54A?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0iI0ZGRiIgZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDptMCAxOGMtNC40MSAwLTgtMy41OS04LThzMy41OS04IDgtOCA4IDMuNTkgOCA4LTMuNTkgOC04IDh6Ii8+PC9zdmc+&logoColor=white" alt="Stable Baselines3"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV"/>
  
  <!-- Control -->
  <img src="https://img.shields.io/badge/PyAutoGUI-2F4F4F?logo=python&logoColor=white" alt="PyAutoGUI"/>

  <!-- Dev -->
  <img src="https://img.shields.io/badge/Pytest-0A9B0A?logo=pytest&logoColor=white" alt="Pytest"/>
  <img src="https://img.shields.io/badge/Ruff-222?logo=ruff&logoColor=white" alt="Ruff"/>
  <img src="https://img.shields.io/badge/Black-000000?logo=python&logoColor=white" alt="Black"/>
</p>

A macOS screen-capture + input-control RL player that allows trained CNN policies (e.g., PPO) to play Mini Motorways by observing the game grid from pixels and acting through mouse clicks/drags.

## Overview

This project provides a complete framework for running reinforcement learning agents on the game Mini Motorways, directly on a macOS desktop. It intelligently captures the game window, processes the visual data in real-time, and translates agent decisions into mouse actions. The system is designed for flexibility, supporting models from popular libraries like Stable-Baselines3 and PyTorch.

### Key Features

- **Direct Screen Capture**: Uses macOS Quartz for efficient, specific window capture without full-screen recording.
- **Ratio-Based Calibration**: A robust two-point calibration system that adapts to window moves and resizes.
- **Precise Mouse Control**: Leverages PyAutoGUI for reliable mouse clicks and drags, with built-in failsafe protection.
- **Broad Model Support**: Load and run policies from Stable-Baselines3 (PPO) and PyTorch.
- **Real-Time Performance**: Achieves a 6-12 FPS agent loop, suitable for real-time gameplay.
- **Comprehensive Logging**: Detailed episode data, including actions and frame hashes, are saved to JSONL files.
- **Safe Dry-Run Mode**: Test your complete setup, calibration, and action mappings without executing real mouse clicks.
- **Simple CLI Interface**: All functionalities are accessible through a user-friendly command-line interface.

### Technology Stack

- **Core Language**: Python
- **Platform**: macOS (Quartz for screen capture)
- **ML/RL**: PyTorch, Stable-Baselines3
- **System Control**: PyAutoGUI
- **Image Processing**: OpenCV, Pillow
- **Development**: Pytest (Testing), Ruff (Linting), Black (Formatting), MyPy (Type Checking)

## Architecture

The project is composed of several modular components:

1.  **Capture Engine**: Handles real-time window capture (`mac_quartz.py`) and image preprocessing (`preprocess.py`) to create observations for the agent.
2.  **Control System**: Manages coordinate mapping from the agent's grid-based view to screen pixels (`mapping.py`) and executes mouse actions (`mouse.py`).
3.  **Policy Loader**: A flexible loader (`loader.py`) that supports both Stable-Baselines3 and PyTorch model formats.
4.  **CLI Application**: The main entry point (`main.py`) that ties all components together and exposes them through a simple command interface.

## Quick Start

### Prerequisites

- **macOS** (Apple Silicon or Intel)
- **Mini Motorways** game installed
- **Python 3.9+**
- **Critical macOS Permissions**: You must grant **Screen Recording** and **Accessibility** permissions to your Terminal or IDE.

### Development Setup

1.  **Clone the repository**
    ```bash
    git clone https://github.com/your-username/motorways-rl.git
    cd motorways-rl
    ```

2.  **Install dependencies**
    ```bash
    pip install -e .[dev]
    ```

3.  **Calibrate the game window**
    Start Mini Motorways, then run the calibration command and follow the on-screen prompts.
    ```bash
    motorways calibrate --grid-h 32 --grid-w 32
    ```

4.  **Run the agent**
    ```bash
    # Test with a dry run (no clicks)
    motorways dry-run --max-steps 50

    # Play with a trained model
    motorways play --model path/to/your/model.zip --max-steps 1000
    ```

## Usage

1.  Launch Mini Motorways.
2.  Use the `motorways calibrate` command once to set up the grid mapping.
3.  Use `motorways play --model <path>` to run your trained agent.

## Development Commands

### Code Quality & Testing
```bash
# Format code
black src tests

# Lint and format
ruff check src tests --fix

# Type checking
mypy src

# Run all tests
pytest
```

## Model Training

This package provides the **inference** engine. To train your own models, you need to create a custom [Gymnasium](https://gymnasium.farama.org/) environment that simulates the game's logic.

The action and observation spaces must match those defined in this project:
-   **Action Space**: See `src/motorways/policy/action_space.py`.
-   **Observation Space**: RGB images (e.g., 128x128 pixels).

An example training setup using Stable-Baselines3:
```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create your custom Mini Motorways simulator environment
env = make_vec_env("MiniMotorways-v0", n_envs=4)

# Train a PPO model
model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)

# Save the model for use with this player
model.save("mini_motorways_ppo.zip")
```

## Project Structure

```
motorways-rl/
├── src/motorways/     # Main application source code
│   ├── app/           # CLI entry point
│   ├── capture/       # Screen capture and preprocessing
│   ├── control/       # Mouse control and coordinate mapping
│   ├── policy/        # RL policy loading and action space
│   ├── config/        # Pydantic schemas and defaults
│   └── utils/         # Logging and permission checks
├── tests/             # Pytest unit and integration tests
├── pyproject.toml     # Project metadata and dependencies
└── README.md          # This file
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
