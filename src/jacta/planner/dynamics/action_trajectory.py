# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import torch

from jacta.planner.core.linear_algebra import einsum_jk_ikl_ijl
from jacta.planner.core.types import ControlType


def create_action_trajectory(
    type: ControlType,
    start_actions: torch.FloatTensor,
    end_actions: torch.FloatTensor,
    trajectory_steps: int,
) -> torch.FloatTensor:
    """Creates a zero- or first-order hold array of action vectors with length trajectory_steps.

    Args:
        type: Create zero- or first-order hold action trajectory.
        start_actions: A (na) array containing the start action vectors of the desired trajectories.
        end_actions: A (na) array containing the end action vectors of the desired trajectories.
            For zero-order hold, only the end action vector will be used and held for the entire trajectory.
            For first-order hold, a linear interpolation between start and end action vector will be created.
        trajectory_steps: The length of the resulting action vector array.

    Returns:
        An action vector array (trajectory_steps, na).

    """
    match type:
        case ControlType.ZERO_ORDER_HOLD:
            action_trajectory = torch.repeat_interleave(
                end_actions.unsqueeze(0), trajectory_steps, dim=0
            )
        case ControlType.FIRST_ORDER_HOLD:
            alpha = (
                torch.linspace(0, 1, steps=trajectory_steps + 1)[1:]
                .unsqueeze(1)
                .to(start_actions.device)
            )
            delta_action = (end_actions - start_actions).unsqueeze(0)
            action_trajectory = start_actions + alpha * delta_action
    return action_trajectory


def create_action_trajectories(
    type: ControlType,
    start_actions: torch.FloatTensor,
    end_actions: torch.FloatTensor,
    trajectory_steps: int,
) -> torch.FloatTensor:
    """Creates a zero- or first-order hold arrays of action vectors with length trajectory_steps.

    Args:
        type: Create zero- or first-order hold action trajectory.
        start_actions: A (n, na) array containing the start action vectors of the desired trajectories.
        end_actions: A (n, na) array containing the end action vectors of the desired trajectories.
            For zero-order hold, only the end action vector will be used and held for the entire trajectory.
            For first-order hold, a linear interpolation between start and end action vector will be created.
        trajectory_steps: The length of the resulting action vector array.

    Returns:
        An action vector array (n, trajectory_steps, na).

    """
    match type:
        case ControlType.ZERO_ORDER_HOLD:
            action_trajectories = torch.repeat_interleave(
                end_actions.unsqueeze(1), trajectory_steps, dim=1
            )
        case ControlType.FIRST_ORDER_HOLD:
            alpha = (
                torch.linspace(0, 1, steps=trajectory_steps + 1)[1:]
                .unsqueeze(1)
                .to(start_actions.device)
            )
            a0 = start_actions.unsqueeze(1)
            a1 = end_actions.unsqueeze(1)
            delta = a1 - a0
            action_trajectories = einsum_jk_ikl_ijl(alpha, delta) + a0
        case _:
            raise ValueError(f"Unsupported ControlType: {type}")

    return action_trajectories
