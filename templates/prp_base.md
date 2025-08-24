name: "Base PRP Template v2 - Context-Rich with Validation Loops"
description: |
Purpose
Template optimized for AI agents to implement features with sufficient context and self-validation capabilities to achieve working code through iterative refinement.

Core Principles

* Context is King: Include ALL necessary documentation, examples, and caveats
* Validation Loops: Provide executable tests/lints the AI can run and fix
* Information Dense: Use keywords and patterns from the codebase
* Progressive Success: Start simple, validate, then enhance
* Global rules: Be sure to follow all rules in CLAUDE.md

---

# Goal

Build a **macOS screen-capture + input-control RL player** for **Mini Motorways** that allows a trained **CNN policy** (e.g., PPO) to observe the game grid from pixels and act through mouse clicks/drags. Deliver a reusable Python package + CLI that:

1. calibrates the grid once,
2. streams window frames at 6–12 FPS,
3. preprocesses to the model’s input,
4. maps model actions → precise screen coordinates,
5. performs reliable clicks/drags,
6. logs episodes and supports evaluation rollouts.

# Why

* **Business value & user impact:** Enables rapid research iterations on city/traffic-building agents without game modding. Works on **M1 macOS** (our machines) and avoids IL2CPP mod complexity.
* **Integration with existing features:** Plugs into our training code (Gymnasium + Stable-Baselines3) and lets us benchmark “sim-trained” policies directly on the real UI.
* **Problems solved / for whom:** Researchers and engineers who need a practical bridge from offline CNN-RL to the live game; streamers and demos that show the agent visibly playing.

# What

**User-visible behavior**

* `motorways play --model path --grid 32x32` starts the agent; first run asks for two-pointer calibration (top-left cell center & bottom-right cell center).
* The agent plays, drawing roads via drags and selecting tools via configurable toolbar coordinates.
* A small overlay/console prints FPS, actions, rewards (if available), and emergency stop instructions.

**Technical requirements**

* macOS **Quartz** window capture (window-only, not full screen).
* **PyAutoGUI** (or CGEvent taps) for mouse events.
* Robust **coordinate mapping** that survives window moves/resizes (store calibration as **ratios** relative to window bounds).
* Deterministic **preprocessing** (resize, normalize, channel order).
* Pluggable **policy loader** (SB3 PPO, SB3-Contrib MaskablePPO, or custom Torch).
* **Fail-safe**: moving mouse to (0,0) aborts loop; keyboard interrupt respected.
* **Rate limiting** & backpressure (TARGET\_FPS configurable).
* Logging (JSONL) of frames hashes, actions, and timestamps.

# Success Criteria

* ✅ Agent loop runs at ≥8 FPS on M1 with 128×128 model input.
* ✅ Calibration survives window moves/resizes (no re-calibration needed unless UI layout changes).
* ✅ End-to-end: model receives frames → outputs actions → correct on-screen clicks/drags observed.
* ✅ All tests, lints, and types pass (see Validation Loop).
* ✅ Repro: `uv run motorways play --dry-run` simulates actions without clicking.
* ✅ Documentation includes setup (Accessibility + Screen Recording permissions).

---

# All Needed Context

## Documentation & References (include these in the context window)

* url: [https://developer.apple.com/documentation/coregraphics/cgwindowlistcreateimage](https://developer.apple.com/documentation/coregraphics/cgwindowlistcreateimage)
  why: CoreGraphics API used to capture a specific window; understand bounds, retina scaling, `kCGWindowImageBoundsIgnoreFraming`.

* url: [https://pyobjc.readthedocs.io/](https://pyobjc.readthedocs.io/)
  why: PyObjC bridge details (`NSBitmapImageRep`, `bytesPerRow`, pixel formats) to convert CGImage → NumPy.

* url: [https://pyautogui.readthedocs.io/en/latest/](https://pyautogui.readthedocs.io/en/latest/)
  why: Mouse move/click/drag, **failsafe**, macOS **Accessibility permission** requirements.

* url: [https://gymnasium.farama.org/](https://gymnasium.farama.org/)
  why: Observation/action conventions; wrappers if we later expose a Gym env wrapper around the live game.

* url: [https://stable-baselines3.readthedocs.io/](https://stable-baselines3.readthedocs.io/)
  why: PPO loading, `MultiInputPolicy`, deterministic prediction.

* url: [https://sb3-contrib.readthedocs.io/](https://sb3-contrib.readthedocs.io/) (MaskablePPO)
  why: Action masking pattern if we later mask illegal cells.

* url: [https://opencv.org/](https://opencv.org/) & [https://docs.opencv.org/](https://docs.opencv.org/)
  why: `cv2.resize`, color order (BGR vs RGB), interpolation modes.

* url: [https://github.com/aleju/mario-ai](https://github.com/aleju/mario-ai)
  why: Pixel-based DQN patterns; replay, reward shaping inspiration.

* url: [https://github.com/PWhiddy/PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments)
  why: Clean **Gym wrapper + SB3** pipeline; emulator→Gym abstraction mirrors our “window→policy” adapter.

* file: scripts/play\_motorways.py (from our prior scaffold)
  why: Baseline example for Quartz capture + calibration + action execution.

* docfile: CLAUDE.md
  why: Global code/style/review rules to follow across the repo.

## Current Codebase tree

(Starting minimal; adapt if you already created files.)

```
.
├── README.md
├── pyproject.toml
├── src/
│   └── placeholder.txt
└── tests/
    └── placeholder.txt
```

## Desired Codebase tree (to be created)

```
.
├── README.md
├── CLAUDE.md
├── pyproject.toml
├── uv.lock
├── src/
│   └── motorways/
│       ├── __init__.py
│       ├── app/
│       │   ├── main.py                 # CLI entrypoints (calibrate/play/dry-run)
│       │   └── overlay.py              # (optional) console/overlay status
│       ├── capture/
│       │   ├── mac_quartz.py           # CGWindowList + NSBitmapImageRep → np.ndarray
│       │   └── preprocess.py           # resize/normalize/channel ops
│       ├── control/
│       │   ├── mapping.py              # grid↔screen; ratios; toolbar coords
│       │   └── mouse.py                # clicks/drags (PyAutoGUI wrappers)
│       ├── policy/
│       │   ├── loader.py               # SB3/Torch loader and predict shim
│       │   └── action_space.py         # decode discrete action → (tool, r, c[, path])
│       ├── config/
│       │   ├── schema.py               # pydantic models (Calibration, Settings)
│       │   └── defaults.py             # sane defaults
│       └── utils/
│           ├── logging.py              # JSONL logs
│           └── permissions.py          # checks for Accessibility/Screen Recording
├── scripts/
│   └── demo.sh
└── tests/
    ├── test_mapping.py
    ├── test_preprocess.py
    ├── test_permissions.py
    └── test_policy_loader.py
```

## Known Gotchas of our codebase & Library Quirks

* **CRITICAL: macOS permissions** — Terminal/IDE must have **Accessibility** (for clicks) and **Screen Recording** (for capture). Programmatically check and fail fast with a clear message.
* **Retina scaling** — Window “bounds” are in **logical points**, pixels can be 2×; always derive crop coordinates from the actual captured image width/height.
* **Window occlusion** — `CGWindowListCreateImage` can return the window even if covered, but some settings cause black frames; advise keeping the window visible.
* **PyAutoGUI** coordinate space\*\* is global; multi-monitor setups mean (0,0) may not be top-left of the internal display—use absolute positions from calibration.
* **Color order** — OpenCV uses BGR; our capturer returns RGB. Be explicit to avoid silent accuracy drops.
* **SB3 predict** — Always run with `deterministic=True` for evaluation; keep a fixed preproc (size/normalize) identical to training.
* **Action timing** — Excessive click rate can cause missed inputs; throttle via TARGET\_FPS and optional short `duration` on moves.

---

# Implementation Blueprint

## Data models and structure (type-safe)

* `Calibration` (pydantic):

  ```python
  class Calibration(BaseModel):
      grid_h: int
      grid_w: int
      x0r: float; y0r: float; x1r: float; y1r: float  # ratios
      toolbar: Dict[str, Tuple[float, float]] = {}    # optional button ratio coords
  ```
* `Settings` (pydantic):

  ```python
  class Settings(BaseModel):
      title_substr: str = "Mini Motorways"
      target_fps: int = 8
      input_size: Tuple[int,int] = (128,128)
      model_path: Path
      dry_run: bool = False
  ```
* `Action` (TypedDict/dataclass):

  ```python
  @dataclass
  class Action:
      type: Literal["click","drag","noop","toolbar"]
      r: Optional[int] = None
      c: Optional[int] = None
      path: Optional[List[Tuple[int,int]]] = None
      tool: Optional[str] = None
  ```

## Tasks (ordered)

**Task 1: CREATE `src/motorways/capture/mac_quartz.py`**

* Implement `find_window(title_substr) -> (window_id, bounds_dict)`
* Implement `grab_window(window_id) -> np.ndarray[H,W,3] (RGB, uint8)`
* GOTCHA: use `NSBitmapImageRep` to get `pixelsWide/High` and `bytesPerRow`; slice `:,:w,:3`.

*Pseudocode*

```python
def grab_window(window_id):
    cgimg = CGWindowListCreateImage(..., window_id, kCGWindowImageBoundsIgnoreFraming)
    rep = NSBitmapImageRep.alloc().initWithCGImage_(cgimg)
    w, h = rep.pixelsWide(), rep.pixelsHigh()
    buf = np.frombuffer(rep.bitmapData(), dtype=np.uint8)
    arr = buf.reshape(h, rep.bytesPerRow()//4, 4)[:,:w,:3].copy()
    return arr  # RGB
```

**Task 2: CREATE `src/motorways/control/mapping.py`**

* `to_screen_center(r, c, win_bounds, cal) -> (x,y)` using **ratios** from calibration.
* `crop_grid(img, win_bounds, cal) -> grid_img` mapping ratios to pixel coords.
* Unit-test both (geometry math is a common bug source).

*Pseudocode*

```python
def to_screen_center(r, c, bounds, cal):
    x0 = bounds['X'] + cal.x0r * bounds['Width']
    y0 = bounds['Y'] + cal.y0r * bounds['Height']
    x1 = bounds['X'] + cal.x1r * bounds['Width']
    y1 = bounds['Y'] + cal.y1r * bounds['Height']
    cw = (x1-x0) / cal.grid_w
    ch = (y1-y0) / cal.grid_h
    return int(x0 + (c+0.5)*cw), int(y0 + (r+0.5)*ch)
```

**Task 3: CREATE `src/motorways/control/mouse.py`**

* Thin wrappers over PyAutoGUI: `click(x,y)`, `drag_path([(x,y),...])`, with **failsafe** and small move durations.
* Provide `dry_run` mode that only logs.

**Task 4: CREATE `src/motorways/config/schema.py` & `defaults.py`**

* Pydantic models for `Calibration` & `Settings`.
* JSON (de)serialization to `~/.motorways/calibration.json`.

**Task 5: CREATE `src/motorways/capture/preprocess.py`**

* Deterministic resize/normalize/transposes to match training.
* Expose `prepare(obs_img, input_size) -> np.ndarray[1,C,H,W]`.

**Task 6: CREATE `src/motorways/policy/loader.py`**

* Load SB3 PPO or Torch model; unify `predict(obs_img) -> Action`.
* CRITICAL: keep device selection explicit (mps/cpu).
* Optional `--mask` path for MaskablePPO.

*Pseudocode*

```python
def load_model(path):
    if path.suffix in (".zip",):
        model = PPO.load(path)
        return lambda obs: decode_action(model.predict(obs, deterministic=True)[0])
    else:
        net = torch.load(path, map_location="cpu")
        net.eval()
        return lambda obs: decode_action(net(torch.from_numpy(obs)).argmax(...))
```

**Task 7: CREATE `src/motorways/app/main.py` (CLI)**

* Commands: `calibrate`, `play`, `dry-run`.
* `calibrate`: prompt top-left & bottom-right by “press Enter while hovering”, save ratios & optional toolbar buttons.
* `play`: loop: capture → crop → preprocess → predict → map → click/drag; throttle by `TARGET_FPS`.
* Log JSONL to `~/.motorways/logs/YYYYMMDD.jsonl`.

**Task 8: Tests**

* `tests/test_mapping.py`: deterministic mapping/cropping given synthetic bounds & ratios.
* `tests/test_preprocess.py`: size & dtype checks.
* `tests/test_permissions.py`: stub returns false when perms missing.
* `tests/test_policy_loader.py`: fake model returning fixed action; ensure decode path works.

**Task 9: DX**

* `README.md`: permission setup, calibration, examples.
* `scripts/demo.sh`: run dry-run; echo instructions.
* Add ruff/mypy configs in `pyproject.toml`.

---

# Integration Points

**CONFIG**

* Add to `pyproject.toml`:

  * ruff & mypy config blocks.
  * console script:

    ```toml
    [project.scripts]
    motorways = "motorways.app.main:cli"
    ```

**ROUTES / SERVICES**

* Not a web service. Provide a simple CLI + logs. (Future: expose a small WebSocket for remote control.)

**LOGGING**

* JSONL with fields: `ts`, `frame_sha1`, `action`, `mouse_xy`, `fps`, `notes`.

---

# Validation Loop

**Level 1: Syntax & Style**

```bash
uv run ruff check src --fix
uv run mypy src
```

**Level 2: Unit Tests**
Create tests as above; run:

```bash
uv run pytest -q
```

Minimal test examples:

```python
# tests/test_mapping.py
from motorways.control.mapping import to_screen_center, crop_grid
from motorways.config.schema import Calibration

def test_center_math():
    bounds = {'X':100,'Y':100,'Width':1000,'Height':800}
    cal = Calibration(grid_h=10, grid_w=10, x0r=0.1, y0r=0.1, x1r=0.9, y1r=0.9)
    x,y = to_screen_center(0,0,bounds,cal)
    assert 180 <= x <= 190 and 140 <= y <= 150
```

**Level 3: Integration Test (dry-run & live)**

```bash
# Dry run (no clicking)
uv run motorways play --model models/ppo.zip --dry-run --grid 32x32

# Live
uv run motorways play --model models/ppo.zip --grid 32x32
# Expect mouse movements in-game; Ctrl+C to stop.
```

Final Validation Checklist

* All tests pass: `uv run pytest -q`
* No linting errors: `uv run ruff check src/`
* No type errors: `uv run mypy src/`
* Manual test successful: `uv run motorways play --dry-run` then live
* Error cases handled (missing permissions, window not found)
* Logs informative (JSONL), not spammy
* README updated with permission setup & calibration steps

---

# Anti-Patterns to Avoid

* ❌ Don’t create new patterns when existing ones work (reuse SB3 loader & our mapping helpers).
* ❌ Don’t skip validation because “it should work”.
* ❌ Don’t ignore failing tests—fix root causes.
* ❌ Don’t use sync sleeps that stall capture excessively; keep loop timing simple and measured.
* ❌ Don’t hardcode coordinates; always store ratios in calibration.
* ❌ Don’t `except Exception:` broadly; raise actionable errors (permissions/window not found).
