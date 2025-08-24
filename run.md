🎯 Quick Start Training

  1. Basic DQN Training

  # Simple training with defaults (100K steps)
  python examples/train_dqn.py

  # Custom training duration
  python examples/train_dqn.py --steps 500000

  # Save to specific location
  python examples/train_dqn.py --steps 100000 --save-path
  models/my_dqn_model.zip

  2. Difficulty-Based Training

  # Easy mode (faster learning, good for testing)
  python examples/train_dqn.py --difficulty easy --steps 50000

  # Normal mode (balanced)
  python examples/train_dqn.py --difficulty normal --steps 100000

  # Hard mode (expert performance)
  python examples/train_dqn.py --difficulty hard --steps 500000

  3. Hardware Optimization

  # Use GPU acceleration (NVIDIA)
  python examples/train_dqn.py --device cuda --steps 200000

  # Use Apple Silicon (M1/M2 Macs)
  python examples/train_dqn.py --device mps --steps 200000

  # Force CPU (any system)
  python examples/train_dqn.py --device cpu --steps 100000

  🔧 Custom Training Parameters

  Full Training Command

  python examples/train_dqn.py \
      --steps 200000 \
      --difficulty normal \
      --grid-h 32 \
      --grid-w 32 \
      --device mps \
      --save-path models/expert_dqn.zip \
      --log-level INFO

  Parameter Explanations

  - --steps: Total training timesteps (50K-1M recommended)
  - --difficulty: Hyperparameter preset (easy/normal/hard)
  - --grid-h/--grid-w: Game grid dimensions
  - --device: Hardware acceleration (auto/cpu/cuda/mps)
  - --save-path: Where to save the trained model
  - --log-level: Logging verbosity (DEBUG/INFO/WARNING)

  🏗️ Environment Integration (Next Step)

  The current training script includes a placeholder environment. To train on
  actual gameplay, you need to implement:

  1. Create Real Environment

  # In examples/train_dqn.py, replace the placeholder with:

  class RealMiniMotorwaysEnv(gym.Env):
      def __init__(self, calibration_path):
          super().__init__()

          # Load calibration
          self.calibration = Calibration.load(calibration_path)

          # Set up window capture
          self.window_id, self.bounds = find_window("Mini Motorways")

          # Action and observation spaces
          self.action_space = spaces.Discrete(get_action_space_size(32, 32))
          self.observation_space = spaces.Box(0, 255, (3, 128, 128),
  dtype=np.uint8)

      def reset(self):
          # Reset game to starting state
          # Capture initial screenshot
          img = grab_window(self.window_id)
          obs = crop_grid(img, self.bounds, self.calibration)
          processed = prepare(obs, (128, 128), normalize=False)
          return processed.squeeze(0), {}

      def step(self, action):
          # Decode and execute action
          decoded_action = decode_action(action, 32, 32)

          # Execute in game
          if decoded_action.type == "click":
              x, y = to_screen_center(decoded_action.r, decoded_action.c,
  self.bounds, self.calibration)
              mouse_click(x, y)
          # ... handle other action types

          # Capture new state
          img = grab_window(self.window_id)
          obs = crop_grid(img, self.bounds, self.calibration)
          processed = prepare(obs, (128, 128), normalize=False)

          # Calculate reward (implement your reward function)
          reward = self._calculate_reward(obs)

          # Check if episode is done
          done = self._is_episode_done(obs)

          return processed.squeeze(0), reward, done, False, {}

  2. Reward Function Design

  def _calculate_reward(self, obs):
      """Calculate reward based on game state."""
      # Example reward components:
      # +1.0 for each car successfully delivered
      # +0.1 for traffic flowing smoothly  
      # -0.5 for traffic jams
      # -10.0 for game over
      # +5.0 for reaching new milestones

      reward = 0.0

      # Analyze screenshot for game state
      # This requires computer vision to detect:
      # - Cars on roads
      # - Traffic flow
      # - Buildings and destinations
      # - Score/milestone indicators

      return reward

  🎮 Training Process

  Training Stages

  1. Calibration (Before training)
  motorways calibrate --output calibration.json
  2. Environment Testing
  # Test with random actions first
  motorways dry-run --calibration calibration.json --max-steps 100
  3. DQN Training
  python examples/train_dqn.py --steps 100000 --difficulty normal
  4. Model Testing
  motorways play --model models/dqn_mini_motorways.zip --max-steps 500

  Training Monitoring

  The training script logs key metrics:
  - Episode Rewards: Total score per game
  - Episode Length: How long the agent survived
  - Exploration Rate: Current epsilon value
  - Loss Values: Q-network training progress

  📈 Hyperparameter Tuning

  DQN Hyperparameters by Difficulty

  Easy Mode (Learning/Debug)
  {
      "learning_rate": 2e-4,        # Faster learning
      "exploration_final_eps": 0.1, # More exploration
      "gamma": 0.95,                # Less future focus
      "buffer_size": 50000,         # Smaller buffer
  }

  Normal Mode (Balanced)
  {
      "learning_rate": 1e-4,        # Standard rate
      "exploration_final_eps": 0.05, # Balanced exploration
      "gamma": 0.99,                # Standard discount
      "buffer_size": 50000,         # Standard buffer
  }

  Hard Mode (Expert)
  {
      "learning_rate": 5e-5,        # Careful learning
      "exploration_final_eps": 0.02, # Minimal exploration
      "gamma": 0.995,               # Strong future focus
      "buffer_size": 100000,        # Large buffer
  }

  Custom Hyperparameters

  from motorways.policy.loader import create_dqn_model

  model = create_dqn_model(
      input_shape=(3, 128, 128),
      action_space_size=2053,
      learning_rate=1e-4,
      buffer_size=50000,
      batch_size=32,
      gamma=0.99,
      target_update_interval=1000,
      exploration_fraction=0.1,
      exploration_initial_eps=1.0,
      exploration_final_eps=0.05,
      device="cuda"
  )

  🏆 Training Tips

  Performance Optimization

  - Start Small: Train on 64x64 images first, then 128x128
  - Curriculum Learning: Begin with easier scenarios
  - Batch Size: Use 32-64 for good GPU utilization
  - Target Updates: 1000-step intervals work well

  Common Issues & Solutions

  Slow Learning
  - Increase learning rate carefully (2e-4 max)
  - Check reward signal strength
  - Ensure diverse training experiences

  Unstable Training
  - Reduce learning rate (5e-5)
  - Increase batch size (64)
  - Check for reward clipping needs

  Poor Exploration
  - Increase exploration_fraction (0.2)
  - Extend epsilon decay schedule
  - Verify action space coverage

  Training Duration Guidelines

  - Testing/Debug: 10K-50K steps
  - Basic Performance: 100K-200K steps
  - Good Performance: 500K-1M steps
  - Expert Performance: 1M+ steps

  🔍 Next Steps

  1. Implement Real Environment: Replace placeholder with actual game
  integration
  2. Design Reward Function: Create rewards that encourage good traffic
  management
  3. Add Evaluation: Regular testing during training
  4. Hyperparameter Search: Use Optuna for optimization
  5. Multi-Map Training: Train on different city layouts

  The foundation is ready - you just need to connect it to the actual Mini
  Motorways game!