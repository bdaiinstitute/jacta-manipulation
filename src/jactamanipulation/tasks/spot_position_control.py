# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np

from jactamanipulation.common.constants import (
    SPOT_STATE_INDS,
    STANDING_HEIGHT,
    STANDING_STOWED_POS,
)
from jactamanipulation.tasks.cost_functions import quadratic_norm
from jactamanipulation.tasks.mujoco_task import MujocoTask
from jactamanipulation.tasks.task import TaskConfig

MODEL_PATH = "dexterity/models/xml/scenes/legacy/spot_position_control.xml"


@dataclass
class SpotPositionControlConfig(TaskConfig):
    """Reward configuration for the Spot simple task"""

    w_standing: float = 100.0
    w_forward: float = 0.0
    w_legs: float = 0.0
    w_arms: float = 1.0
    default_command: Optional[np.ndarray] = STANDING_STOWED_POS


class SpotPositionControl(MujocoTask[SpotPositionControlConfig]):
    """Defines the Spot standing up task."""

    def __init__(self) -> None:
        super().__init__(MODEL_PATH)
        self.goal_state = np.concatenate(
            (np.array([0.0, 0.0, STANDING_HEIGHT, 1.0, 0.0, 0.0, 0.0]), STANDING_STOWED_POS)
        )
        self.reset()

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: SpotPositionControlConfig,
    ) -> np.ndarray:
        """Implements the Spot reward to have it stand up and go to a goal.

        Maps a list of states, a list of control, to a batch of rewards (summed over time) for each rollout.

        Spot has four main reward terms:
            * `position_rew`, penalizing the distance between the Spot body position and the final position
            * `orientation_rew`, penalizing the distance between the Spot body orientation and the final orientation
            * `leg_rew`, penalizing the difference between Spot leg angles and the final leg angles
            * `arm_rew`, penalizing the difference between the Spot arm angles and the final arm angles
        """
        batch_size = states.shape[0]

        body_pos_error = (
            states[:, -1, SPOT_STATE_INDS.body_slice]
            - self.goal_state[..., SPOT_STATE_INDS.body_slice]
            + states[:, -1, SPOT_STATE_INDS.body_vel_slice]
        )
        # TODO (@bhung) fix the obviously incorrect error in the quaternion
        orientation_error = self.goal_state[..., SPOT_STATE_INDS.quat_slice] - states[:, -1, SPOT_STATE_INDS.quat_slice]

        legs_error = (
            self.goal_state[..., SPOT_STATE_INDS.legs_slice]
            - states[:, -1, SPOT_STATE_INDS.legs_slice]
            + states[:, -1, SPOT_STATE_INDS.legs_vel_slice]
        )
        arms_error = (
            self.goal_state[..., SPOT_STATE_INDS.arms_slice]
            - states[:, -1, SPOT_STATE_INDS.arms_slice]
            + states[:, -1, SPOT_STATE_INDS.arms_vel_slice]
        )

        position_rew = -config.w_standing * quadratic_norm(body_pos_error)
        orientation_rew = -config.w_forward * quadratic_norm(orientation_error)
        legs_rew = -config.w_legs * quadratic_norm(legs_error)
        arms_rew = -config.w_arms * quadratic_norm(arms_error)

        total_rew = position_rew + orientation_rew + legs_rew + arms_rew

        assert position_rew.shape == (batch_size,)
        assert orientation_rew.shape == (batch_size,)
        assert legs_rew.shape == (batch_size,)
        assert arms_rew.shape == (batch_size,)

        return total_rew

    def reset(self) -> None:
        """Resets the model to a slightly random state"""
        self.data.qpos = self.goal_state
        self.data.qvel = 1e-2 * np.random.randn(self.model.nv)
        self.data.ctrl = STANDING_STOWED_POS
        mujoco.mj_forward(self.model, self.data)
