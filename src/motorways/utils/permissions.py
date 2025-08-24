"""macOS permissions checking for Screen Recording and Accessibility."""

import logging

logger = logging.getLogger(__name__)


def check_screen_recording_permission() -> bool:
    """Check if Screen Recording permission is granted.

    Returns:
        True if permission is granted, False otherwise
    """
    try:
        # Try pyautogui first as it's simpler and more reliable
        try:
            import pyautogui
            # Try a simple screenshot test
            screenshot = pyautogui.screenshot()
            if screenshot is not None:
                logger.info("Screen Recording permission confirmed via pyautogui")
                return True
        except Exception as e:
            logger.debug(f"PyAutoGUI screenshot failed: {e}")

        # Fallback to window capture method
        from motorways.capture.mac_quartz import find_window, grab_window

        # Find any window (preferably Finder which should always exist)
        window_id, bounds = find_window("Finder")

        if window_id is None:
            # Try to find any window
            from Quartz import (
                CGWindowListCopyWindowInfo,
                kCGNullWindowID,
                kCGWindowListOptionOnScreenOnly,
            )
            window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)

            if not window_list:
                logger.warning("No windows found for permission test")
                return False

            # Use first available window and get its bounds
            first_window = window_list[0]
            window_id = first_window.get('kCGWindowNumber')
            bounds = first_window.get('kCGWindowBounds', {})

        if window_id is None:
            logger.warning("Could not find suitable window for permission test")
            return False

        # Try to capture the window with bounds
        try:
            img = grab_window(window_id, bounds)
            if img is not None and img.size > 0:
                logger.info("Screen Recording permission confirmed")
                return True
            else:
                logger.warning("Screen capture returned empty image - permission likely denied")
                return False
        except Exception as e:
            logger.warning(f"Screen capture failed - permission likely denied: {e}")
            return False

    except Exception as e:
        logger.error(f"Screen Recording permission check failed: {e}")
        return False


def check_accessibility_permission() -> bool:
    """Check if Accessibility permission is granted.

    Returns:
        True if permission is granted, False otherwise
    """
    try:
        import pyautogui

        # Try to get mouse position - this requires Accessibility permission
        try:
            pos = pyautogui.position()
            if pos is not None:
                logger.info("Accessibility permission confirmed")
                return True
            else:
                logger.warning("Could not get mouse position - permission likely denied")
                return False
        except Exception as e:
            logger.warning(f"Mouse position check failed - permission likely denied: {e}")
            return False

    except Exception as e:
        logger.error(f"Accessibility permission check failed: {e}")
        return False


def check_all_permissions() -> dict[str, bool]:
    """Check all required permissions.

    Returns:
        Dictionary with permission status for each requirement
    """
    permissions = {
        "screen_recording": check_screen_recording_permission(),
        "accessibility": check_accessibility_permission()
    }

    logger.info(f"Permission status: {permissions}")
    return permissions


def get_permission_instructions() -> dict[str, str]:
    """Get instructions for granting permissions.

    Returns:
        Dictionary with permission names and setup instructions
    """
    return {
        "screen_recording": """
To grant Screen Recording permission:
1. Open System Preferences/Settings > Security & Privacy > Privacy
2. Click "Screen Recording" in the left sidebar
3. Click the lock to make changes
4. Check the box next to your Terminal/IDE application
5. Restart your Terminal/IDE application
""",
        "accessibility": """
To grant Accessibility permission:
1. Open System Preferences/Settings > Security & Privacy > Privacy
2. Click "Accessibility" in the left sidebar
3. Click the lock to make changes
4. Check the box next to your Terminal/IDE application
5. Restart your Terminal/IDE application
"""
    }


def validate_permissions_or_exit() -> None:
    """Check permissions and exit with instructions if not granted.

    Raises:
        SystemExit: If required permissions are not granted
    """
    permissions = check_all_permissions()
    missing_permissions = [name for name, granted in permissions.items() if not granted]

    if missing_permissions:
        logger.error(f"Missing required permissions: {missing_permissions}")

        instructions = get_permission_instructions()
        print("\n🚨 Missing Required Permissions 🚨\n")

        for perm in missing_permissions:
            print(f"❌ {perm.replace('_', ' ').title()}")
            print(instructions[perm])
            print()

        print("Please grant the required permissions and try again.")
        print("Note: You may need to restart your Terminal/IDE after granting permissions.\n")

        raise SystemExit(1)

    logger.info("✅ All required permissions granted")


def prompt_for_permission_grant() -> bool:
    """Prompt user to grant permissions and wait for confirmation.

    Returns:
        True if user confirms permissions are granted
    """
    permissions = check_all_permissions()
    missing_permissions = [name for name, granted in permissions.items() if not granted]

    if not missing_permissions:
        return True

    instructions = get_permission_instructions()
    print("\n🚨 Required Permissions Not Granted 🚨\n")

    for perm in missing_permissions:
        print(f"❌ {perm.replace('_', ' ').title()}")
        print(instructions[perm])
        print()

    print("After granting permissions:")
    print("1. Restart your Terminal/IDE")
    print("2. Run this command again")

    response = input("\nHave you granted the permissions and restarted? (y/N): ").strip().lower()

    if response in ['y', 'yes']:
        # Re-check permissions
        new_permissions = check_all_permissions()
        still_missing = [name for name, granted in new_permissions.items() if not granted]

        if still_missing:
            print(f"\n❌ Still missing permissions: {still_missing}")
            print("Please ensure you've granted all permissions and restarted your Terminal/IDE.")
            return False
        else:
            print("\n✅ All permissions confirmed!")
            return True
    else:
        print("\nPlease grant the required permissions before proceeding.")
        return False
