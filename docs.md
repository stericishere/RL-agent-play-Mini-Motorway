Your environment is now structurally complete!
  Here's how to finish implementing it:

  2. Calibrate Your Setup

  First, run the calibration to set up coordinates:

  # Start Mini Motorways and run calibration
  motorways calibrate --output calibration.json

  # Follow the prompts to click on grid corners and 
  toolbar buttons

  3. Test the Environment

  Test your environment with a simple script:

  # Create test_env.py
  from examples.train_dqn import
  create_training_environment

  # Test environment creation
  try:
      env = create_training_environment(calibration_
  path="calibration.json")
      print("✅ Environment created successfully!")

      # Test reset
      obs, info = env.reset()
      print(f"✅ Reset successful, observation 
  shape: {obs.shape}")

      # Test a few random actions
      for i in range(5):
          action = env.action_space.sample()
          obs, reward, done, truncated, info =
  env.step(action)
          print(f"Step {i}: action={action}, 
  reward={reward:.3f}, done={done}")

          if done:
              obs, info = env.reset()

  except Exception as e:
      print(f"❌ Error: {e}")

  4. Enhance the Reward Function

  The current reward function is basic. Improve it
  by detecting game elements:

  def _calculate_reward(self, obs):
      """Enhanced reward function with computer 
  vision."""
      import cv2

      # Convert to different color spaces for 
  analysis
      hsv = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
      gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

      # Detect different game elements
      rewards = {
          'survival': 0.1,  # Base survival reward
          'traffic_flow': 0.0,
          'construction': 0.0,
          'efficiency': 0.0
      }

      # 1. Traffic flow detection (green = good, red
   = bad)
      green_mask = cv2.inRange(hsv, (40, 50, 50),
  (80, 255, 255))
      red_mask = cv2.inRange(hsv, (0, 50, 50), (20,
  255, 255))

      green_pixels = np.sum(green_mask > 0)
      red_pixels = np.sum(red_mask > 0)

      if green_pixels > red_pixels:
          rewards['traffic_flow'] = 0.5  # Good 
  traffic flow
      elif red_pixels > green_pixels * 2:
          rewards['traffic_flow'] = -0.3  # Traffic 
  jam penalty

      # 2. Construction activity (new 
  roads/buildings)
      # Look for construction indicators, building 
  animations
      edges = cv2.Canny(gray, 50, 150)
      activity_level = np.sum(edges > 0) /
  (obs.shape[0] * obs.shape[1])

      if activity_level > 0.1:  # High activity 
  threshold
          rewards['construction'] = 0.2

      # 3. Efficiency (cars reaching destinations)
      # This requires detecting moving cars and 
  destination buildings
      # You'd implement optical flow or template 
  matching here

      total_reward = sum(rewards.values())
      self.episode_reward += total_reward

      return total_reward

  5. Game Over Detection

  Implement proper episode termination:

  def _is_episode_done(self, obs):
      """Enhanced episode termination detection."""
      import cv2

      # Check step limit
      if self.current_step >= self.max_steps:
          return True

      # Detect game over screen by looking for 
  specific UI elements
      gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

      # Method 1: Look for "Game Over" text or 
  restart button
      # You'd train a template matcher or use OCR 
  here

      # Method 2: Detect if screen is static 
  (paused/game over)
      if hasattr(self, 'previous_obs'):
          diff = cv2.absdiff(gray,
  self.previous_obs)
          movement = np.sum(diff > 30) / diff.size

          if movement < 0.001:  # Very little change
              self.static_frames = getattr(self,
  'static_frames', 0) + 1
              if self.static_frames > 30:  # 3 
  seconds of no movement
                  logger.info("Episode ended: game 
  appears static/paused")
                  return True
          else:
              self.static_frames = 0

      self.previous_obs = gray.copy()

      # Method 3: Detect score screen or menu
      # Look for UI elements that indicate end of 
  game

      return False

  6. Game State Management

  Add proper game reset functionality:

  def reset(self, seed=None, options=None):
      """Reset with proper game state management."""
      self.current_step = 0
      self.episode_reward = 0

      # Game reset options:
      # Option 1: Press R key to restart
      import pyautogui
      pyautogui.press('r')
      time.sleep(2)  # Wait for restart

      # Option 2: Click restart button (if you 
  calibrated it)
      # if 'restart' in self.calibration.toolbar:
      #     x, y = get_toolbar_coord('restart', 
  self.bounds, self.calibration)
      #     mouse_click(x, y)
      #     time.sleep(3)  # Wait for new game to 
  load

      # Option 3: Navigate through menus
      # This requires more complex state machine

      # Capture fresh state after reset
      img = grab_window(self.window_id)
      obs = crop_grid(img, self.bounds,
  self.calibration)
      processed = prepare(obs, (128, 128),
  normalize=False)

      info = {"step": self.current_step}
      return processed.squeeze(0), info

  7. Training with Real Environment

  Now you can train with the real game:

  # Make sure Mini Motorways is running and 
  calibrated
  python examples/train_dqn.py \
      --steps 10000 \
      --calibration calibration.json \
      --difficulty easy \
      --device auto \
      --save-path models/dqn_real_game.zip

  8. Advanced Features

  Once basic training works, add these enhancements:

  Multi-Threading

  # Separate game interaction from training
  import threading
  from queue import Queue

  class ThreadedEnvironment:
      def __init__(self, base_env):
          self.env = base_env
          self.action_queue = Queue()
          self.obs_queue = Queue()
          self.game_thread =
  threading.Thread(target=self._game_loop)
          self.game_thread.daemon = True
          self.game_thread.start()

  Performance Monitoring

  # Add metrics tracking
  class MetricsTracker:
      def __init__(self):
          self.episode_scores = []
          self.survival_times = []
          self.traffic_efficiency = []

      def log_episode(self, score, steps, 
  efficiency):
          self.episode_scores.append(score)
          self.survival_times.append(steps)
          self.traffic_efficiency.append(efficiency)

  Curriculum Learning

  # Gradually increase difficulty
  class CurriculumManager:
      def __init__(self):
          self.current_level = 1
          self.success_threshold = 0.7

      def should_advance(self, recent_success_rate):
          return recent_success_rate >
  self.success_threshold

  🚀 Ready to Train!

  Your environment is now ready for real training!
  The key steps:

  1. ✅ Environment Structure: Complete with action
  execution, rewards, episode management
  2. 🎯 Next: Calibrate your specific setup and test
   the environment
  3. 🔧 Enhance: Improve reward function based on
  Mini Motorways gameplay
  4. 🚀 Train: Start with short episodes and scale
  up

  Start with short training runs (1K-10K steps) to
  validate everything works, then scale to longer
  training sessions!



  🔧 Fix Screen Recording Permission

  Step 1: Grant Screen Recording Permission

  1. Open System Settings (macOS 13+) or System 
  Preferences (macOS 12 and below)
  2. Navigate to Privacy & Security → Privacy → 
  Screen Recording
    - On newer macOS: System Settings → Privacy & 
  Security → Screen Recording
    - On older macOS: System Preferences → Security 
  & Privacy → Privacy → Screen Recording
  3. Click the lock icon to make changes (you'll
  need to enter your password)
  4. Add your Terminal application:
    - If using Terminal: Check the box next to
  "Terminal"
    - If using iTerm2: Check the box next to "iTerm"
    - If using VS Code integrated terminal: Check
  the box next to "Code" or "Visual Studio Code"
  5. Restart your terminal application completely
  (quit and reopen)

  Step 2: Verify Permission

  After granting permission and restarting your
  terminal, test it:

  # Test the permission check
  motorways calibrate --output calibration.json

  Step 3: Alternative Permission Check

  If you want to test screen capture independently:

  # Test screen capture directly
  python -c "
  from motorways.capture.mac_quartz import 
  find_window, grab_window
  import numpy as np

  # Try to find any window
  window_id, bounds = find_window('Finder')
  if window_id:
      print('✅ Found window:', bounds)
      try:
          img = grab_window(window_id)
          print('✅ Screen capture successful, image
   shape:', img.shape)
      except Exception as e:
          print('❌ Screen capture failed:', e)
  else:
      print('ℹ️  No Finder window found, but 
  that\\'s normal')
  "

  🎮 Start Mini Motorways Before Calibration

  Once screen recording permission is granted:

  1. Launch Mini Motorways
  2. Start a new game (or have a game running)
  3. Run calibration:

  motorways calibrate --output calibration.json

  The calibration process will:
  1. Find the Mini Motorways window
  2. Ask you to click on grid corners
  3. Optionally calibrate toolbar buttons
  4. Save the coordinates to calibration.json

  🔍 If Still Having Issues

  If you continue to get permission errors after
  following the steps above:

  Check Current Permissions

  # Check what the system thinks about permissions
  python -c "
  from motorways.utils.permissions import 
  check_all_permissions
  perms = check_all_permissions()
  print('Screen Recording:', '✅' if 
  perms['screen_recording'] else '❌')
  print('Accessibility:', '✅' if 
  perms['accessibility'] else '❌')
  "

  Reset Privacy Settings (if needed)

  If permissions are still not working:

  1. Open Terminal and run:
  sudo tccutil reset ScreenCapture

  2. Restart your Mac (this ensures all permission
  changes take effect)
  3. Re-grant the permissions following Step 1 above

  Alternative Terminal Applications

  If your current terminal doesn't work, try:
  - Built-in Terminal.app: Usually works best with
  macOS permissions
  - iTerm2: Popular alternative with good permission
   support

  🚀 Once Permission is Granted

  After you get screen recording permission working:

  1. Calibrate your setup:
  motorways calibrate --output calibration.json

  2. Test with dry run:
  motorways dry-run --calibration calibration.json
  --max-steps 10

  3. Train your DQN model:
  python examples/train_dqn.py \
      --steps 1000 \
      --calibration calibration.json \
      --difficulty easy