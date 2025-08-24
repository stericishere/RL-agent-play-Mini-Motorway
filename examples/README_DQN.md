# DQN Training for Mini Motorways

This directory contains examples and utilities for training Deep Q-Network (DQN) models to play Mini Motorways.

## Overview

The system now supports DQN training using Stable-Baselines3, specifically optimized for the Mini Motorways game environment. DQN is well-suited for this discrete action space game where agents need to learn optimal road placement strategies.

## Features

### DQN Model Support
- **Stable-Baselines3 Integration**: Full support for SB3 DQN models
- **Automatic Model Detection**: Automatically detects and loads DQN vs PPO models
- **CNN Policy**: Uses convolutional neural networks optimized for image input
- **Hyperparameter Optimization**: Pre-configured hyperparameters for Mini Motorways

### Key DQN Advantages for Mini Motorways
- **Discrete Actions**: Perfect fit for click/drag/toolbar actions
- **Experience Replay**: Learns from past gameplay experiences efficiently
- **Epsilon-Greedy Exploration**: Balances exploration vs exploitation during training
- **Target Networks**: Stable learning with periodic target network updates

## Quick Start

### 1. Install Dependencies

```bash
# Install the package with DQN support
pip install -e .

# Or install Stable-Baselines3 separately
pip install stable-baselines3[extra]
```

### 2. Train a DQN Model

```bash
# Basic training with default settings
python examples/train_dqn.py --steps 100000

# Advanced training with custom settings  
python examples/train_dqn.py \
    --steps 500000 \
    --difficulty hard \
    --grid-h 32 \
    --grid-w 32 \
    --device cuda \
    --save-path models/dqn_expert.zip
```

### 3. Use Trained DQN Model

```bash
# Run the trained DQN model
motorways play --model models/dqn_mini_motorways.zip

# The system will automatically detect it's a DQN model and configure accordingly
```

## DQN Hyperparameters

### Recommended Settings by Difficulty

**Easy (Learning/Debug)**
- Learning Rate: 2e-4
- Exploration Final Epsilon: 0.1
- Gamma (Discount): 0.95
- Buffer Size: 50,000

**Normal (Balanced)**
- Learning Rate: 1e-4  
- Exploration Final Epsilon: 0.05
- Gamma (Discount): 0.99
- Buffer Size: 50,000

**Hard (Expert Performance)**
- Learning Rate: 5e-5
- Exploration Final Epsilon: 0.02
- Gamma (Discount): 0.995
- Buffer Size: 100,000

## Action Space

The DQN model operates on a discrete action space optimized for Mini Motorways:

- **No-op**: Do nothing (action 0)
- **Click Actions**: Click at grid positions (1 to grid_h×grid_w)
- **Drag Actions**: Start drag from grid positions (next grid_h×grid_w actions)
- **Toolbar Actions**: Select tools (road, bridge, roundabout, traffic_light, motorway, house)

For a 32×32 grid: 2,053 total discrete actions (1 + 1024 + 1024 + 6)

## Training Tips

### Environment Setup
1. **Game Integration**: Connect to actual Mini Motorways game or simulator
2. **Reward Design**: Define rewards based on traffic flow, city growth, survival time
3. **State Representation**: Use processed RGB screenshots (3×128×128)
4. **Episode Length**: Typical games last 500-2000 steps

### Training Strategies
1. **Start Simple**: Begin with smaller grids and easier scenarios
2. **Curriculum Learning**: Gradually increase difficulty during training
3. **Exploration Schedule**: Decay epsilon from 1.0 to 0.05 over training
4. **Evaluation**: Regular evaluation episodes with deterministic policy

## Implementation Example

```python
from motorways.policy.loader import create_dqn_model, get_recommended_dqn_hyperparameters

# Get optimized hyperparameters
params = get_recommended_dqn_hyperparameters(grid_size=(32, 32), difficulty="normal")

# Create DQN model
model = create_dqn_model(
    input_shape=params["input_shape"],
    action_space_size=params["action_space_size"],
    **params
)

# Train on your environment
model.learn(total_timesteps=100000)

# Save for inference
model.save("models/dqn_motorways.zip")
```

## Performance Monitoring

Monitor these metrics during training:

- **Episode Reward**: Total reward per game episode
- **Episode Length**: Steps survived before game over
- **Exploration Rate**: Current epsilon value
- **Loss Values**: Q-network and target network losses
- **Traffic Flow**: Cars successfully routed (game-specific)

## Troubleshooting

### Common Issues

**Slow Learning**
- Increase learning rate (be careful of instability)  
- Reduce target update interval
- Check reward signal strength

**Unstable Training**
- Reduce learning rate
- Increase batch size
- Check for reward clipping

**Poor Exploration**
- Increase exploration_fraction
- Adjust epsilon decay schedule
- Verify action space coverage

### Device Optimization

**CPU Training**
- Good for prototyping and small models
- Set `device="cpu"` explicitly

**GPU Training (CUDA)**
- Significantly faster for CNN policies
- Requires CUDA-compatible GPU
- Set `device="cuda"`

**Apple Silicon (MPS)**
- Native support for M1/M2 Macs
- Set `device="mps"`
- Fallback to CPU if MPS unavailable

## Next Steps

1. **Implement Environment**: Create gymnasium environment for Mini Motorways
2. **Reward Engineering**: Design effective reward functions
3. **Hyperparameter Tuning**: Use tools like Optuna for optimization
4. **Multi-Agent**: Explore multi-agent scenarios with multiple cities
5. **Transfer Learning**: Train on different maps and game modes

For more details, see the main project documentation and the `motorways.policy.loader` module.