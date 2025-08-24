## FEATURE:

**Mini Motorways RL Player (macOS, Screen-Capture + Mouse Control)**
A runnable agent loop that:

* captures the Mini Motorways window via Quartz,
* crops/normalizes the grid,
* feeds frames to a CNN policy (e.g., SB3 PPO),
* converts actions → clicks/drags,
* logs runs and supports dry-run.

## EXAMPLES:

> All examples live in `examples/` and can be run with `uv run python <file>.py`.

1. **`00_find_window.py`**

   * Prints discovered window id/bounds for `"Mini Motorways"`.
   * Verifies Screen Recording permission is working (non-empty frame shape).

2. **`01_calibrate.py`**

   * Interactive: hover **top-left cell center**, press Enter; then **bottom-right cell center**, press Enter.
   * Saves `~/.motorways/calibration.json` with ratio-based anchors (survives window moves/resizes).

3. **`02_capture_and_crop.py`**

   * Streams window frames (RGB) and shows only the cropped grid region (derived from calibration).
   * Confirms Retina math is correct (no shearing / off-by-one edges).

4. **`03_random_click_agent.py`** *(dry run by default)*

   * Random grid clicks at 6–8 FPS, prints mapped `(r,c) -> (x,y)` screen coords.
   * `--live` flag performs real clicks (requires Accessibility permission).

5. **`04_policy_infer_sb3.py`**

   * Loads `models/ppo_mini_motorways.zip` (SB3).
   * Preprocesses crop to `128×128` (RGB→CHW, /255.0), calls `model.predict(..., deterministic=True)`.
   * Decodes discrete action → `{tool, r, c}`; draws short road segments via drag when action encodes a path.

6. **`05_toolbar_demo.py`**

   * Demonstrates selecting **road/bridge/roundabout** toolbar buttons using ratio coordinates stored in calibration (e.g., `toolbar.road = [0.12, 0.93]`).
   * Clicks button then executes a two-cell drag.

## DOCUMENTATION:

* **CoreGraphics / Quartz** – `CGWindowListCreateImage` (window-only capture), image bounds flags
  [https://developer.apple.com/documentation/coregraphics/cgwindowlistcreateimage](https://developer.apple.com/documentation/coregraphics/cgwindowlistcreateimage)

* **PyObjC** – bridging `CGImage` → `NSBitmapImageRep` → NumPy (pay attention to `bytesPerRow`)
  [https://pyobjc.readthedocs.io/](https://pyobjc.readthedocs.io/)

* **PyAutoGUI** – mouse move/click/drag, failsafe, macOS Accessibility permissions
  [https://pyautogui.readthedocs.io/en/latest/](https://pyautogui.readthedocs.io/en/latest/)

* **Gymnasium** – observation/action conventions (if wrapping the live loop later)
  [https://gymnasium.farama.org/](https://gymnasium.farama.org/)

* **Stable-Baselines3 (SB3)** – loading PPO, `MultiInputPolicy`, deterministic prediction
  [https://stable-baselines3.readthedocs.io/](https://stable-baselines3.readthedocs.io/)

* **SB3-Contrib (optional)** – MaskablePPO pattern (if we add action masking)
  [https://sb3-contrib.readthedocs.io/](https://sb3-contrib.readthedocs.io/)

* **Reference repos** (patterns for pixel RL & clean wrappers):

  * Mario (pixel DQN): [https://github.com/aleju/mario-ai](https://github.com/aleju/mario-ai)
  * Pokémon Red Experiments (Gym wrapper + SB3): [https://github.com/PWhiddy/PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments)

## OTHER CONSIDERATIONS:

* **Permissions are non-negotiable**

  * Grant **Screen Recording** (capture) and **Accessibility** (clicks) to your terminal/IDE. If clicks “do nothing,” re-grant Accessibility after OS updates.

* **Retina / coordinate math**

  * Window bounds are in **logical points**; captured image is **pixels** (often 2×). Always compute crops in **image-pixel space**; store calibration as **ratios** of window width/height.

* **Calibration durability**

  * Calibration is robust to window moves/resizes but **not** to UI layout changes (zoom, theme). Deleting `~/.motorways/calibration.json` forces a quick re-calibration.

* **Color order & preprocessing**

  * Capturer returns **RGB**; OpenCV defaults to **BGR**. Be explicit when resizing/normalizing so it matches training.

* **Action pacing**

  * Over-clicking can drop inputs. Limit loop to 6–12 FPS; add a tiny `moveTo(..., duration=0.01)` before drags to improve reliability.

* **Multi-monitor setups**

  * Global screen (0,0) may be off-screen depending on arrangement. Use absolute coords from calibration, not assumptions about origin.

* **Window occlusion**

  * Quartz usually captures occluded windows, but some setups can yield black frames. Keep the game visible in front during runs.

* **Determinism for eval**

  * Use `deterministic=True` on SB3 `predict`; fix preprocessing dimensions and normalization identical to training.

* **Safety / abort**

  * Enable PyAutoGUI **failsafe** (slam mouse to top-left corner to stop). Also handle `KeyboardInterrupt` cleanly.

* **Future path (optional)**

  * If we outgrow screen-scrape, plan migration to a Unity mod (BepInEx IL2CPP macOS arm64) for direct state/actions.
