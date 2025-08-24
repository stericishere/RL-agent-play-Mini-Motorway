"""Action space definitions and decoding for Mini Motorways RL agent."""

import logging

import numpy as np

from motorways.config.schema import Action

logger = logging.getLogger(__name__)


def decode_action(action_value: int, grid_h: int, grid_w: int) -> Action:
    """Decode discrete action value to Action object.

    Action space encoding:
    - 0: No-op
    - 1 to grid_h * grid_w: Click at grid cell (row, col)
    - grid_h * grid_w + 1 to 2 * grid_h * grid_w: Start drag from cell
    - Next ranges: Toolbar selections (road, bridge, roundabout, etc.)

    Args:
        action_value: Discrete action integer from model
        grid_h: Grid height in cells
        grid_w: Grid width in cells

    Returns:
        Action object representing the decoded action

    Raises:
        ValueError: If action value is invalid
    """
    total_grid_cells = grid_h * grid_w

    # No-op action
    if action_value == 0:
        return Action(type="noop")

    # Click actions: 1 to grid_h * grid_w
    elif 1 <= action_value <= total_grid_cells:
        cell_idx = action_value - 1
        r = cell_idx // grid_w
        c = cell_idx % grid_w
        logger.debug(f"Decoded click action: cell {cell_idx} -> grid ({r}, {c})")
        return Action(type="click", r=r, c=c)

    # Simple drag actions: next grid_h * grid_w actions
    # Each represents starting a 2-cell horizontal drag from that position
    elif total_grid_cells < action_value <= 2 * total_grid_cells:
        cell_idx = action_value - total_grid_cells - 1
        start_r = cell_idx // grid_w
        start_c = cell_idx % grid_w

        # Create a simple 2-cell horizontal drag
        if start_c + 1 < grid_w:  # Can drag right
            path = [(start_r, start_c), (start_r, start_c + 1)]
        else:  # Drag down if can't drag right
            if start_r + 1 < grid_h:
                path = [(start_r, start_c), (start_r + 1, start_c)]
            else:
                # Fallback to single click if can't drag anywhere
                return Action(type="click", r=start_r, c=start_c)

        logger.debug(f"Decoded drag action: start cell {cell_idx} -> path {path}")
        return Action(type="drag", path=path)

    # Toolbar actions
    elif 2 * total_grid_cells < action_value <= 2 * total_grid_cells + 6:
        toolbar_tools = ["road", "bridge", "roundabout", "traffic_light", "motorway", "house"]
        tool_idx = action_value - 2 * total_grid_cells - 1

        if 0 <= tool_idx < len(toolbar_tools):
            tool_name = toolbar_tools[tool_idx]
            logger.debug(f"Decoded toolbar action: {tool_name}")
            return Action(type="toolbar", tool=tool_name)
        else:
            logger.warning(f"Invalid toolbar action index: {tool_idx}")
            return Action(type="noop")

    else:
        logger.warning(f"Invalid action value: {action_value}")
        return Action(type="noop")


def get_action_space_size(grid_h: int, grid_w: int) -> int:
    """Get total action space size for given grid dimensions.

    Args:
        grid_h: Grid height in cells
        grid_w: Grid width in cells

    Returns:
        Total number of discrete actions available
    """
    total_grid_cells = grid_h * grid_w

    # 1 no-op + grid_cells clicks + grid_cells drags + 6 toolbar actions
    total_actions = 1 + total_grid_cells + total_grid_cells + 6

    logger.debug(f"Action space size for {grid_h}x{grid_w} grid: {total_actions}")
    return total_actions


def encode_action(action: Action, grid_h: int, grid_w: int) -> int:
    """Encode Action object to discrete action value.

    Args:
        action: Action object to encode
        grid_h: Grid height in cells
        grid_w: Grid width in cells

    Returns:
        Discrete action integer

    Raises:
        ValueError: If action cannot be encoded
    """
    total_grid_cells = grid_h * grid_w

    if action.type == "noop":
        return 0

    elif action.type == "click":
        if action.r is None or action.c is None:
            raise ValueError("Click action missing coordinates")

        if not (0 <= action.r < grid_h and 0 <= action.c < grid_w):
            raise ValueError(f"Click coordinates ({action.r}, {action.c}) out of bounds")

        cell_idx = action.r * grid_w + action.c
        return cell_idx + 1

    elif action.type == "drag":
        if not action.path or len(action.path) < 2:
            raise ValueError("Drag action missing valid path")

        # Use starting cell for encoding
        start_r, start_c = action.path[0]
        if not (0 <= start_r < grid_h and 0 <= start_c < grid_w):
            raise ValueError(f"Drag start coordinates ({start_r}, {start_c}) out of bounds")

        cell_idx = start_r * grid_w + start_c
        return total_grid_cells + cell_idx + 1

    elif action.type == "toolbar":
        if action.tool is None:
            raise ValueError("Toolbar action missing tool name")

        toolbar_tools = ["road", "bridge", "roundabout", "traffic_light", "motorway", "house"]
        try:
            tool_idx = toolbar_tools.index(action.tool)
            return 2 * total_grid_cells + tool_idx + 1
        except ValueError:
            raise ValueError(f"Unknown toolbar tool: {action.tool}")

    else:
        raise ValueError(f"Unknown action type: {action.type}")


def create_action_mask(grid_h: int, grid_w: int, invalid_cells: list[tuple[int, int]] = None) -> np.ndarray:
    """Create action mask for invalid actions.

    Args:
        grid_h: Grid height in cells
        grid_w: Grid width in cells
        invalid_cells: List of (r, c) coordinates that are invalid for placement

    Returns:
        Boolean mask array where True means action is valid
    """
    action_space_size = get_action_space_size(grid_h, grid_w)
    mask = np.ones(action_space_size, dtype=bool)

    if invalid_cells is None:
        return mask

    total_grid_cells = grid_h * grid_w

    # Mark invalid click actions
    for r, c in invalid_cells:
        if 0 <= r < grid_h and 0 <= c < grid_w:
            cell_idx = r * grid_w + c
            click_action_idx = cell_idx + 1
            drag_action_idx = total_grid_cells + cell_idx + 1

            # Mask both click and drag actions for invalid cells
            mask[click_action_idx] = False
            mask[drag_action_idx] = False

    logger.debug(f"Created action mask with {mask.sum()} valid actions out of {action_space_size}")
    return mask


def sample_random_action(grid_h: int, grid_w: int, action_mask: np.ndarray = None) -> Action:
    """Sample a random valid action.

    Args:
        grid_h: Grid height in cells
        grid_w: Grid width in cells
        action_mask: Optional mask of valid actions

    Returns:
        Random Action object
    """
    action_space_size = get_action_space_size(grid_h, grid_w)

    if action_mask is not None:
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0:
            logger.warning("No valid actions available, returning noop")
            return Action(type="noop")
        action_value = np.random.choice(valid_actions)
    else:
        action_value = np.random.randint(0, action_space_size)

    return decode_action(action_value, grid_h, grid_w)
