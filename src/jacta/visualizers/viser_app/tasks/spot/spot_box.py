# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from jacta.visualizers.viser_app.constants import (
    ARM_UNSTOWED_POS,
    STANDING_UNSTOWED_POS,
)
from jacta.visualizers.viser_app.path_utils import DATA_PATH, MODEL_PATH
from jacta.visualizers.viser_app.tasks.spot_base import (
    DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME,
    GOAL_POSITIONS,
    SpotBase,
    SpotBaseConfig,
)

Z_AXIS = np.array([0.0, 0.0, 1.0])
RESET_OBJECT_POSE = np.array([3, 0, 0.275, 1, 0, 0, 0])
# annulus object position sampling
RADIUS_MIN = 1.0
RADIUS_MAX = 2.0


@dataclass
class SpotBoxConfig(SpotBaseConfig):
    """Config for the spot box manipulation task."""

    default_command: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 0.0, *ARM_UNSTOWED_POS])
    )
    goal_position: np.ndarray = GOAL_POSITIONS().black_cross
    w_orientation: float = 15.0
    w_torso_proximity: float = 0.1
    w_gripper_proximity: float = 4.0
    orientation_threshold: float = 0.5


class SpotBox(SpotBase[SpotBoxConfig]):
    """Task getting Spot to move a box to a desired goal location."""

    def __init__(self) -> None:
        self.model_filename = str(MODEL_PATH / "xml/spot_box.xml")
        self.policy_filename = str(DATA_PATH / "policies/xinghao_policy_v1.onnx")
        super().__init__(self.model_filename, self.policy_filename)
        self.command_mask = np.arange(0, 10)

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: SpotBoxConfig,
        additional_info: dict[str, Any],
    ) -> np.ndarray:
        """Reward function for the Spot box moving task."""
        batch_size = states.shape[0]

        body_height = states[..., 2]
        body_pos = states[..., 0:3]
        object_pos = states[..., 26:29]

        object_y_axis = sensors[..., 3:6]
        torso_to_object = sensors[..., 6:9]

        # Check if any state in the rollout has spot fallen
        spot_fallen_reward = -config.fall_penalty * (
            body_height <= config.spot_fallen_threshold
        ).any(axis=-1)

        # Compute l2 distance from object pos. to goal.
        goal_reward = -config.w_goal * np.linalg.norm(
            object_pos - np.array(config.goal_position)[None, None], axis=-1
        ).mean(-1)

        box_orientation_reward = -config.w_orientation * np.abs(
            np.dot(object_y_axis, Z_AXIS) > config.orientation_threshold
        ).sum(axis=-1)

        # Compute l2 distance from torso pos. to object pos.
        torso_proximity_reward = config.w_torso_proximity * np.linalg.norm(
            body_pos - object_pos, axis=-1
        ).mean(-1)

        # Compute l2 distance from torso pos. to object pos.
        gripper_proximity_reward = -config.w_gripper_proximity * np.linalg.norm(
            torso_to_object,
            axis=-1,
        ).mean(-1)

        # Compute a velocity penalty to prefer small velocity commands.
        controls_reward = -config.w_controls * np.linalg.norm(controls, axis=-1).mean(
            -1
        )

        assert spot_fallen_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert box_orientation_reward.shape == (batch_size,)
        assert torso_proximity_reward.shape == (batch_size,)
        assert gripper_proximity_reward.shape == (batch_size,)
        assert controls_reward.shape == (batch_size,)

        return (
            spot_fallen_reward
            + goal_reward
            + box_orientation_reward
            + torso_proximity_reward
            + gripper_proximity_reward
            + controls_reward
        )

    def reset(self) -> None:
        """Reset function for the spot box manipulation task ."""
        radius = RADIUS_MIN + (RADIUS_MAX - RADIUS_MIN) * np.random.rand()
        theta = 2 * np.pi * np.random.rand()
        object_pos = [radius * np.cos(theta), radius * np.cos(theta)]
        reset_object_pose = np.array([*object_pos, 0.275, 1, 0, 0, 0])

        self.data.qpos = np.array(
            [0, 0, 0.52, 1, 0, 0, 0, *STANDING_UNSTOWED_POS, *reset_object_pose]
        )
        self.data.qpos[:2] += np.random.randn(2)
        self.data.qpos[-7:-5] += np.random.randn(2)
        self.data.qvel = np.zeros_like(self.data.qvel)
        mujoco.mj_forward(self.model, self.data)
        self.last_policy_outputs = np.copy(self.initial_policy_output)
        self._additional_info["last_policy_output"] = self.initial_policy_output
        self.cutoff_time = DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME

    @property
    def nu(self) -> int:
        """Number of controls for this task."""
        return 10

    @property
    def actuator_ctrlrange(self) -> np.ndarray:
        """Control bounds for this task."""
        lower_bound = np.concatenate(
            (
                -0.7 * np.ones(3),
                ARM_UNSTOWED_POS - np.array([0.7, 1.3, 0.7, 0.7, 0.7, 0.7, 0]),
            )
        )
        upper_bound = np.concatenate(
            (
                0.7 * np.ones(3),
                ARM_UNSTOWED_POS + np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0]),
            )
        )
        return np.stack([lower_bound, upper_bound], axis=-1)
