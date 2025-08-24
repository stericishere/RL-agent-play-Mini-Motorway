"""Main CLI application for Mini Motorways RL Player."""

import logging
import time
from pathlib import Path
from typing import Optional

import click

from motorways.capture.mac_quartz import find_window, get_window_bounds, grab_window
from motorways.capture.preprocess import prepare
from motorways.config.defaults import (
    DEFAULT_INPUT_SIZE,
    DEFAULT_TARGET_FPS,
    DEFAULT_TITLE_SUBSTR,
    get_default_calibration_path,
    get_default_log_path,
)
from motorways.config.schema import Action, Calibration
from motorways.control.mapping import (
    crop_grid,
    get_toolbar_coord,
    get_upgrade_coord,
    to_screen_center,
    validate_calibration,
)
from motorways.control.mouse import click as mouse_click
from motorways.control.mouse import configure_failsafe, drag_path, get_mouse_position
from motorways.policy.loader import create_random_policy, load_model
from motorways.utils.logging import JSONLLogger, setup_logging
from motorways.utils.permissions import validate_permissions_or_exit

logger = logging.getLogger(__name__)


@click.group()
@click.option('--log-level', default='INFO', help='Logging level')
@click.option('--log-file', type=click.Path(), help='Log file path')
def cli(log_level: str, log_file: Optional[str]) -> None:
    """Mini Motorways RL Player - macOS screen capture + mouse control agent."""
    setup_logging(log_level, Path(log_file) if log_file else None)


@cli.command()
@click.option('--title', default=DEFAULT_TITLE_SUBSTR, help='Window title substring')
@click.option('--grid-h', default=32, help='Grid height in cells')
@click.option('--grid-w', default=32, help='Grid width in cells')
@click.option('--output', type=click.Path(), help='Calibration output file')
def calibrate(title: str, grid_h: int, grid_w: int, output: Optional[str]) -> None:
    """Calibrate grid boundaries and toolbar positions."""
    logger.info("Starting calibration process...")

    # Check permissions
    validate_permissions_or_exit()

    # Find window
    window_id, bounds = find_window(title)
    if window_id is None:
        click.echo(f"❌ Window containing '{title}' not found")
        click.echo("Make sure Mini Motorways is running and visible")
        raise click.Abort()

    click.echo(f"✅ Found window: {bounds}")

    # Interactive calibration
    click.echo("\n📍 Grid Calibration")
    click.echo("Position your mouse over the CENTER of the TOP-LEFT grid cell")
    input("Press Enter when ready...")

    x0, y0 = get_mouse_position()
    x0r = (x0 - bounds['X']) / bounds['Width']
    y0r = (y0 - bounds['Y']) / bounds['Height']

    click.echo(f"Top-left recorded: ({x0}, {y0}) -> ratio ({x0r:.4f}, {y0r:.4f})")

    click.echo("\nPosition your mouse over the CENTER of the BOTTOM-RIGHT grid cell")
    input("Press Enter when ready...")

    x1, y1 = get_mouse_position()
    x1r = (x1 - bounds['X']) / bounds['Width']
    y1r = (y1 - bounds['Y']) / bounds['Height']

    click.echo(f"Bottom-right recorded: ({x1}, {y1}) -> ratio ({x1r:.4f}, {y1r:.4f})")

    # Create calibration object
    try:
        calibration = Calibration(
            grid_h=grid_h,
            grid_w=grid_w,
            x0r=x0r,
            y0r=y0r,
            x1r=x1r,
            y1r=y1r
        )
    except Exception as e:
        click.echo(f"❌ Invalid calibration: {e}")
        raise click.Abort() from e

    # Validate calibration
    if not validate_calibration(calibration, bounds):
        click.echo("❌ Calibration validation failed")
        raise click.Abort()

    # Enhanced toolbar calibration
    if click.confirm("\n🛠️  Calibrate toolbar region (recommended for dynamic positioning)?"):
        click.echo("Position mouse over the TOP-LEFT corner of the toolbar area")
        input("Press Enter when ready...")
        
        tx0, ty0 = get_mouse_position()
        tx0r = (tx0 - bounds['X']) / bounds['Width']
        ty0r = (ty0 - bounds['Y']) / bounds['Height']
        
        click.echo("Position mouse over the BOTTOM-RIGHT corner of the toolbar area")  
        input("Press Enter when ready...")
        
        tx1, ty1 = get_mouse_position()
        tx1r = (tx1 - bounds['X']) / bounds['Width']
        ty1r = (ty1 - bounds['Y']) / bounds['Height']
        
        calibration.toolbar_region = {
            'x0r': tx0r, 'y0r': ty0r, 'x1r': tx1r, 'y1r': ty1r
        }
        click.echo(f"Toolbar region: ({tx0}, {ty0}) -> ({tx1}, {ty1})")

    # Optional specific toolbar button calibration (legacy support)
    if click.confirm("\n📍 Also calibrate specific toolbar buttons? (optional - for fallback)"):
        toolbar_buttons = ["road", "bridge", "roundabout", "traffic_light"]

        for button in toolbar_buttons:
            if click.confirm(f"Calibrate {button} button?"):
                click.echo(f"Position mouse over the {button} button")
                input("Press Enter when ready...")

                bx, by = get_mouse_position()
                bxr = (bx - bounds['X']) / bounds['Width']
                byr = (by - bounds['Y']) / bounds['Height']

                calibration.toolbar[button] = (bxr, byr)
                click.echo(
                    f"{button} button: ({bx}, {by}) -> "
                    f"ratio ({bxr:.4f}, {byr:.4f})"
                )

    # Upgrade selection calibration
    if click.confirm("\n🚀 Calibrate upgrade selection area?"):
        click.echo("Please trigger an upgrade screen in the game first...")
        click.echo("(Play until you get an upgrade choice screen)")
        input("Press Enter when upgrade screen is visible...")
        
        click.echo("Position mouse over the TOP-LEFT corner of the upgrade selection area")
        input("Press Enter when ready...")
        
        ux0, uy0 = get_mouse_position()
        ux0r = (ux0 - bounds['X']) / bounds['Width']
        uy0r = (uy0 - bounds['Y']) / bounds['Height']
        
        click.echo("Position mouse over the BOTTOM-RIGHT corner of the upgrade selection area")
        input("Press Enter when ready...")
        
        ux1, uy1 = get_mouse_position()
        ux1r = (ux1 - bounds['X']) / bounds['Width']
        uy1r = (uy1 - bounds['Y']) / bounds['Height']
        
        calibration.upgrade_region = {
            'x0r': ux0r, 'y0r': uy0r, 'x1r': ux1r, 'y1r': uy1r
        }
        
        # Calibrate individual upgrade options
        upgrade_positions = ["left", "middle", "right"]
        for position in upgrade_positions:
            if click.confirm(f"Calibrate {position} upgrade option?"):
                click.echo(f"Position mouse over the {position} upgrade choice")
                input("Press Enter when ready...")
                
                upx, upy = get_mouse_position()
                upxr = (upx - bounds['X']) / bounds['Width']
                upyr = (upy - bounds['Y']) / bounds['Height']
                
                calibration.upgrades[position] = (upxr, upyr)
                click.echo(f"{position} upgrade: ({upx}, {upy}) -> ratio ({upxr:.4f}, {upyr:.4f})")
        
        click.echo(f"Upgrade region: ({ux0}, {uy0}) -> ({ux1}, {uy1})")

    # Save calibration
    cal_path = Path(output) if output else get_default_calibration_path()
    calibration.save(cal_path)

    click.echo(f"\n✅ Calibration saved to {cal_path}")

    # Test calibration
    if click.confirm("Test calibration with a few sample points?"):
        click.echo("\nTesting calibration...")
        test_points = [(0, 0), (0, grid_w-1), (grid_h-1, 0), (grid_h-1, grid_w-1)]

        for r, c in test_points:
            screen_x, screen_y = to_screen_center(r, c, bounds, calibration)
            click.echo(f"Grid ({r}, {c}) -> Screen ({screen_x}, {screen_y})")

    click.echo("\n🎉 Calibration complete!")


@cli.command()
@click.option(
    '--model', required=True, type=click.Path(exists=True), help='Model file path'
)
@click.option('--title', default=DEFAULT_TITLE_SUBSTR, help='Window title substring')
@click.option('--fps', default=DEFAULT_TARGET_FPS, help='Target FPS')
@click.option(
    '--input-size',
    default=f"{DEFAULT_INPUT_SIZE[0]}x{DEFAULT_INPUT_SIZE[1]}",
    help='Input size (WxH)'
)
@click.option('--calibration', type=click.Path(), help='Calibration file path')
@click.option('--dry-run', is_flag=True, help='Dry run mode (no actual clicks)')
@click.option('--max-steps', default=1000, help='Maximum steps to run')
def play(
    model: str,
    title: str,
    fps: int,
    input_size: str,
    calibration: Optional[str],
    dry_run: bool,
    max_steps: int
) -> None:
    """Run the RL agent playing Mini Motorways."""
    logger.info("Starting agent play session...")

    # Check permissions (unless dry run)
    if not dry_run:
        validate_permissions_or_exit()

    # Parse input size
    try:
        width, height = map(int, input_size.split('x'))
        input_size_tuple = (width, height)
    except ValueError:
        click.echo(
            f"❌ Invalid input size format: {input_size}. Use WxH (e.g., 128x128)"
        )
        raise click.Abort() from None

    # Load calibration
    cal_path = Path(calibration) if calibration else get_default_calibration_path()
    if not cal_path.exists():
        click.echo(f"❌ Calibration file not found: {cal_path}")
        click.echo("Run 'motorways calibrate' first")
        raise click.Abort()

    try:
        cal = Calibration.load(cal_path)
        logger.info(f"Loaded calibration: {cal.grid_h}x{cal.grid_w} grid")
    except Exception as e:
        click.echo(f"❌ Failed to load calibration: {e}")
        raise click.Abort() from e

    # Find window
    window_id, bounds = find_window(title)
    if window_id is None:
        click.echo(f"❌ Window containing '{title}' not found")
        raise click.Abort()

    # Validate calibration against current window
    if not validate_calibration(cal, bounds):
        click.echo("❌ Calibration invalid for current window")
        click.echo("Try running 'motorways calibrate' again")
        raise click.Abort()

    # Load model
    try:
        predict_fn = load_model(
            Path(model),
            device="cpu",  # TODO: Add device option
            grid_h=cal.grid_h,
            grid_w=cal.grid_w
        )
        logger.info(f"Loaded model: {model}")
    except Exception as e:
        click.echo(f"❌ Failed to load model: {e}")
        raise click.Abort() from e

    # Setup logging
    log_path = get_default_log_path() / f"play_{int(time.time())}.jsonl"
    jsonl_logger = JSONLLogger(log_path)

    # Configure mouse safety
    configure_failsafe(not dry_run)  # Disable failsafe in dry run

    # Main agent loop
    click.echo(f"\n🤖 Starting agent loop (max {max_steps} steps)")
    click.echo(f"📊 Target FPS: {fps}")
    click.echo(f"📝 Logging to: {log_path}")
    if dry_run:
        click.echo("🧪 DRY RUN MODE - No actual clicks")

    jsonl_logger.log_event("session_start", {
        "model": model,
        "calibration": str(cal_path),
        "window_title": title,
        "target_fps": fps,
        "dry_run": dry_run,
        "max_steps": max_steps
    })

    try:
        step = 0
        frame_time = 1.0 / fps

        while step < max_steps:
            loop_start = time.time()

            # Capture window
            try:
                # Update window bounds in case window moved
                current_bounds = get_window_bounds(window_id)
                if current_bounds:
                    bounds = current_bounds

                img = grab_window(window_id)
                grid_img = crop_grid(img, bounds, cal)

            except Exception as e:
                logger.error(f"Frame capture failed: {e}")
                jsonl_logger.log_event(
                    "error", {"step": step, "error": "capture_failed"}, str(e)
                )
                time.sleep(frame_time)
                continue

            # Preprocess for model
            try:
                obs = prepare(grid_img, input_size_tuple, normalize=True)
            except Exception as e:
                logger.error(f"Preprocessing failed: {e}")
                time.sleep(frame_time)
                continue

            # Get action from model
            try:
                action = predict_fn(obs)
                logger.debug(f"Step {step}: {action.type}")
            except Exception as e:
                logger.error(f"Model prediction failed: {e}")
                action = Action(type="noop")

            # Execute action
            mouse_xy = None
            try:
                if (action.type == "click" and
                        action.r is not None and action.c is not None):
                    x, y = to_screen_center(action.r, action.c, bounds, cal)
                    mouse_click(x, y, dry_run=dry_run)
                    mouse_xy = (x, y)

                elif action.type == "drag" and action.path:
                    # Convert grid path to screen coordinates
                    screen_path = []
                    for r, c in action.path:
                        x, y = to_screen_center(r, c, bounds, cal)
                        screen_path.append((x, y))
                    drag_path(screen_path, dry_run=dry_run)
                    mouse_xy = screen_path[-1] if screen_path else None

                elif action.type == "toolbar" and action.tool:
                    if action.tool in cal.toolbar:
                        x, y = get_toolbar_coord(action.tool, bounds, cal)
                        mouse_click(x, y, dry_run=dry_run)
                        mouse_xy = (x, y)
                    else:
                        logger.warning(f"Toolbar button '{action.tool}' not calibrated")

                elif action.type == "upgrade" and action.upgrade_choice:
                    if action.upgrade_choice in cal.upgrades:
                        x, y = get_upgrade_coord(action.upgrade_choice, bounds, cal)
                        mouse_click(x, y, dry_run=dry_run)
                        mouse_xy = (x, y)
                        logger.info(f"Selected upgrade: {action.upgrade_choice}")
                    else:
                        logger.warning(f"Upgrade choice '{action.upgrade_choice}' not calibrated")

            except Exception as e:
                logger.error(f"Action execution failed: {e}")
                jsonl_logger.log_event(
                    "error", {"step": step, "error": "action_failed"}, str(e)
                )

            # Calculate actual FPS
            loop_time = time.time() - loop_start
            actual_fps = 1.0 / loop_time if loop_time > 0 else 0

            # Log action
            jsonl_logger.log_action(
                action=action,
                frame=grid_img,
                mouse_xy=mouse_xy,
                fps=actual_fps
            )

            # Print status
            if step % 10 == 0:
                click.echo(
                    f"Step {step}/{max_steps} | FPS: {actual_fps:.1f} | "
                    f"Action: {action.type}"
                )

            # Sleep to maintain target FPS
            sleep_time = frame_time - loop_time
            if sleep_time > 0:
                time.sleep(sleep_time)

            step += 1

    except KeyboardInterrupt:
        click.echo("\n⏹️  Stopped by user")
    except Exception as e:
        logger.error(f"Agent loop failed: {e}")
        click.echo(f"❌ Agent loop failed: {e}")
    finally:
        jsonl_logger.log_event("session_end", {"final_step": step})
        click.echo(f"\n📊 Session complete: {step} steps")

        # Show log stats
        stats = jsonl_logger.get_log_stats()
        click.echo(
            f"📝 Log entries: {stats['entries']}, "
            f"Size: {stats.get('size_mb', 0):.1f}MB"
        )


@cli.command()
@click.option('--title', default=DEFAULT_TITLE_SUBSTR, help='Window title substring')
@click.option('--calibration', type=click.Path(), help='Calibration file path')
@click.option('--fps', default=6, help='Target FPS for random actions')
@click.option('--max-steps', default=50, help='Maximum steps to run')
def dry_run(title: str, calibration: Optional[str], fps: int, max_steps: int) -> None:
    """Test setup with random actions (no model required)."""
    logger.info("Starting dry run with random actions...")

    # Load calibration
    cal_path = Path(calibration) if calibration else get_default_calibration_path()
    if not cal_path.exists():
        click.echo(f"❌ Calibration file not found: {cal_path}")
        click.echo("Run 'motorways calibrate' first")
        raise click.Abort()

    try:
        cal = Calibration.load(cal_path)
    except Exception as e:
        click.echo(f"❌ Failed to load calibration: {e}")
        raise click.Abort() from e

    # Find window
    window_id, bounds = find_window(title)
    if window_id is None:
        click.echo(f"❌ Window containing '{title}' not found")
        raise click.Abort()

    # Create random policy
    predict_fn = create_random_policy(cal.grid_h, cal.grid_w)

    # Setup logging
    log_path = get_default_log_path() / f"dry_run_{int(time.time())}.jsonl"
    jsonl_logger = JSONLLogger(log_path)

    click.echo("\n🎲 Starting dry run with random actions")
    click.echo(f"📊 Target FPS: {fps}")
    click.echo(f"📝 Logging to: {log_path}")

    try:
        step = 0
        frame_time = 1.0 / fps

        while step < max_steps:
            loop_start = time.time()

            # Capture and process frame
            try:
                img = grab_window(window_id)
                grid_img = crop_grid(img, bounds, cal)
            except Exception as e:
                logger.error(f"Capture failed: {e}")
                time.sleep(frame_time)
                continue

            # Get random action
            action = predict_fn(None)  # Random policy doesn't need observation

            # Log what we would do
            if action.type == "click":
                x, y = to_screen_center(action.r, action.c, bounds, cal)
                click.echo(
                    f"Step {step}: Click at grid ({action.r}, {action.c}) -> "
                    f"screen ({x}, {y})"
                )
            elif action.type == "drag":
                click.echo(f"Step {step}: Drag with {len(action.path)} points")
            elif action.type == "toolbar":
                click.echo(f"Step {step}: Select toolbar '{action.tool}'")
            else:
                click.echo(f"Step {step}: {action.type}")

            # Log action
            jsonl_logger.log_action(action=action, frame=grid_img)

            # Sleep
            sleep_time = frame_time - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

            step += 1

    except KeyboardInterrupt:
        click.echo("\n⏹️  Stopped by user")

    click.echo(f"\n✅ Dry run complete: {step} steps")


if __name__ == '__main__':
    cli()
