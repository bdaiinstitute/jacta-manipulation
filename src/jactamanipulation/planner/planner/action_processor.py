# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
from dataclasses import dataclass
from typing import Optional

import torch
from benedict import benedict
from torch import FloatTensor, IntTensor

from dexterity.jacta_planner.dynamics.action_trajectory import create_action_trajectories
from dexterity.jacta_planner.planner.clipping_method import clip_actions
from dexterity.jacta_planner.planner.linear_algebra import project_vectors_on_eigenspace
from dexterity.jacta_planner.planner.types import (
    ACTION_TYPE_DIRECTIONAL_MAP,
    ActionMode,
    ActionType,
    ControlType,
)


@dataclass
class ActionProcessor:
    """ActionProcessor"""

    params: benedict
    actuated_pos: IntTensor

    def get_start_actions(
        self,
        current_actuated_states: Optional[FloatTensor] = None,
        previous_end_actions: Optional[FloatTensor] = None,
    ) -> FloatTensor:
        """Compute start action or previous end action

        Compute start action based on current state or previous end action
        depending on the action_start_mode parameter.
        """
        match self.params.action_start_mode:
            case ActionMode.RELATIVE_TO_CURRENT_STATE:
                assert current_actuated_states is not None
                actions = current_actuated_states
            case ActionMode.RELATIVE_TO_PREVIOUS_END_ACTION:
                assert previous_end_actions is not None
                actions = previous_end_actions
            case _:
                print("Select a valid ActionMode for the params.action_start_mode.")
                raise (NotImplementedError)
        actions = clip_actions(actions, self.params)

        return actions

    def get_end_actions(
        self,
        relative_actions: FloatTensor,
        action_type: Optional[ActionType] = None,
        current_actuated_states: Optional[FloatTensor] = None,
        previous_end_actions: Optional[FloatTensor] = None,
    ) -> FloatTensor:
        """Compute end action

        Compute end action as absolute action or relative to current state or previous end action.
        The computation depends on the action_end_mode parameter.
        """
        match self.params.action_end_mode:
            case ActionMode.RELATIVE_TO_CURRENT_STATE:
                assert current_actuated_states is not None
                actions = current_actuated_states + relative_actions
            case ActionMode.RELATIVE_TO_PREVIOUS_END_ACTION:
                assert previous_end_actions is not None
                actions = previous_end_actions + relative_actions
            case ActionMode.ABSOLUTE_ACTION:
                actions = relative_actions
            case _:
                print("Select a valid ActionMode for the params.action_end_mode.")
                raise (NotImplementedError)
        actions = clip_actions(actions, self.params)

        is_directional = ACTION_TYPE_DIRECTIONAL_MAP.get(action_type, True)
        if not is_directional and self.params.using_eigenspaces:
            actions = project_vectors_on_eigenspace(actions, self.params.orthonormal_basis)

        return actions

    def get_action_trajectories(
        self,
        start_actions: FloatTensor,
        end_actions: FloatTensor,
    ) -> FloatTensor:
        """Create and return action trajectories

        Args:
            start_actions (FloatTensor): Array containing the start action vectors of the desired trajectories
            end_actions (FloatTensor): Array containing the end action vectors of the desired trajectories

        Returns:
            FloatTensor: An action vector array (n, trajectory_steps, na)
        """
        action_trajectories = create_action_trajectories(
            type=self.params.control_type,
            start_actions=start_actions,
            end_actions=end_actions,
            trajectory_steps=self.params.num_substeps,
        )
        return action_trajectories

    def get_actuated_states(self, current_states: FloatTensor) -> FloatTensor:
        """Return actuated states

        Args:
            current_states (FloatTensor): Current states

        Returns:
            FloatTensor: Actuated
        """
        return current_states[:, self.actuated_pos]

    def __call__(
        self,
        relative_actions: FloatTensor,
        action_type: ActionType | None = None,
        current_states: FloatTensor | None = None,
        previous_end_actions: FloatTensor | None = None,
    ) -> FloatTensor:
        """Makes the class Callable. On call, return action trajectories

        Args:
            relative_actions (FloatTensor): Relative actions
            action_type (ActionType | None, optional): Action type. Defaults to None.
            current_states (FloatTensor | None, optional): Current states. Defaults to None.
            previous_end_actions (FloatTensor | None, optional): Previous end actions. Defaults to None.

        Returns:
            FloatTensor: _description_
        """
        current_actuated_states = self.get_actuated_states(current_states)
        start_actions = self.get_start_actions(current_actuated_states, previous_end_actions)
        end_actions = self.get_end_actions(relative_actions, action_type, current_actuated_states, previous_end_actions)
        action_trajectories = self.get_action_trajectories(start_actions, end_actions)
        return start_actions, end_actions, action_trajectories


@dataclass
class SpotFloatingActionProcessor(ActionProcessor):
    """SpotFloatingActionProcessor"""

    base_action_ixs: slice = slice(0, 3)  # noqa: RUF009
    arm_action_ixs: slice = slice(3, 10)  # noqa: RUF009

    def get_end_actions(
        self,
        relative_actions: FloatTensor,
        action_type: Optional[ActionType] = None,
        current_actuated_states: Optional[FloatTensor] = None,
        previous_end_actions: Optional[FloatTensor] = None,
    ) -> FloatTensor:
        """Computes end action for Spot with floating base.

        Given relative actions in polar coordinates it computes the absolute action
        for the floating base motion.
        """
        assert current_actuated_states is not None
        relative_actions = relative_actions.clone()  # prevents in-place modification
        # convert actions to robot frame (egocentric)
        base_actions = self.base_action_to_egocentric(relative_actions, current_actuated_states)
        # set base actions in the relative actions tensor
        relative_actions[:, self.base_action_ixs] = base_actions
        return super().get_end_actions(
            relative_actions=relative_actions,
            action_type=action_type,
            current_actuated_states=current_actuated_states,
            previous_end_actions=previous_end_actions,
        )

    def base_action_to_egocentric(
        self, relative_actions: FloatTensor, current_actuated_states: FloatTensor
    ) -> FloatTensor:
        """Convert robot floating base world frame actions to robot frame "egocentric" actions."""
        base_actions = relative_actions[:, self.base_action_ixs]
        longitudinal = base_actions[:, 0]
        lateral = base_actions[:, 1]
        delta_yaw = base_actions[:, 2]
        yaw = current_actuated_states[:, self.base_action_ixs][:, 2]

        # scale the backward motion
        relative_actions[longitudinal < 0, 0] *= self.params.action_backward_speed_mag

        # convert to robot frame
        base_action = torch.stack(
            [
                longitudinal * torch.cos(yaw) - lateral * torch.sin(yaw),
                longitudinal * torch.sin(yaw) + lateral * torch.cos(yaw),
                delta_yaw,
            ],
            dim=1,
        )
        return self.clip_base_actions(base_action)

    def clip_base_actions(self, base_action: FloatTensor) -> FloatTensor:
        """Clips based on clipping type

        Args:
            base_action (FloatTensor): Base action

        Returns:
            FloatTensor: Clipped base action
        """
        clip_params = benedict(
            action_bound_lower=self.params.action_bound_lower[self.base_action_ixs],
            action_bound_upper=self.params.action_bound_upper[self.base_action_ixs],
            clipping_type=self.params.clipping_type,
        )
        base_action = clip_actions(base_action, clip_params)
        return base_action


@dataclass
class SpotWholebodyActionProcessor(ActionProcessor):
    """SpotWholebodyActionProcessor"""

    base_action_ixs: slice = slice(0, 3)  # noqa: RUF009
    arm_action_ixs: slice = slice(3, 10)  # noqa: RUF009
    arm_state_ixs = slice(12, 12 + 7)
    """
    Action processor for the Spot robot with wholebody control.
    Uses vx, vy, vtheta to control the locomotion policy powered plant which handles the leg control.
    """

    def get_actuated_states(self, current_states: FloatTensor) -> FloatTensor:
        """Gets actuated states from the current states

        Args:
            current_states (FloatTensor): Current states

        Returns:
            FloatTensor: Actuated states
        """
        current_actuated_states = super().get_actuated_states(current_states)
        n_states = current_actuated_states.shape[0]
        # using dummy base state to support clipping/ranging functionality for high level commands.
        dummy_base_state = torch.zeros((n_states, 3), device=current_states.device)
        arm_state = current_actuated_states[:, self.arm_state_ixs]
        return torch.cat([dummy_base_state, arm_state], dim=1)

    def get_start_actions(
        self,
        current_actuated_states: Optional[FloatTensor] = None,
        previous_end_actions: Optional[FloatTensor] = None,
    ) -> FloatTensor:
        """Get start actions for the ARM."""
        assert current_actuated_states is not None
        n_states = current_actuated_states.shape[0]
        dummy_base_actions = torch.zeros((n_states, 3), device=current_actuated_states.device)
        arm_actions = super().get_start_actions(
            current_actuated_states=current_actuated_states,
            previous_end_actions=previous_end_actions,
        )[:, self.arm_action_ixs]
        return torch.cat([dummy_base_actions, arm_actions], dim=1)

    def get_action_trajectories(
        self,
        start_actions: FloatTensor,
        end_actions: FloatTensor,
    ) -> FloatTensor:
        """Separately create trajectories for the base and the arm actions.

        Base actions are zero-order held, while arm actions are parameterized.
        """
        arm_start_actions = start_actions[:, self.arm_action_ixs]
        base_end_actions = end_actions[:, self.base_action_ixs]
        arm_end_actions = end_actions[:, self.arm_action_ixs]
        base_trajectory = create_action_trajectories(
            type=ControlType.ZERO_ORDER_HOLD,
            start_actions=None,
            end_actions=base_end_actions,
            trajectory_steps=self.params.num_substeps,
        )
        arm_trajectory = create_action_trajectories(
            type=self.params.control_type,
            start_actions=arm_start_actions,
            end_actions=arm_end_actions,
            trajectory_steps=self.params.num_substeps,
        )
        return torch.cat([base_trajectory, arm_trajectory], dim=-1)
