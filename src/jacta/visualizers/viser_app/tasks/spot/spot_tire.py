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

TIRE_HEIGHT: float = 0.515
TIRE_GOAL = np.array([1.63, -0.53, TIRE_HEIGHT / 2])
RESET_OBJECT_POSE = np.array([3, 0, 0.275, 1, 0, 0, 0])
Z_AXIS = np.array([0.0, 0.0, 1.0])


@dataclass
class SpotTireConfig(SpotBaseConfig):
    """Config for the spot tire manipulation task."""

    default_command: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 0.0, *ARM_UNSTOWED_POS])
    )
    goal_position: np.ndarray = GOAL_POSITIONS().black_cross
    goal_tire_pos: np.ndarray = TIRE_GOAL
    fall_penalty: float = 5000.0
    tire_fallen_threshold: float = 0.1
    w_torso_proximity: float = 1.0
    torso_goal_offset: float = 1.0
    w_gripper_proximity: float = 1.0
    gripper_goal_offset: float = 0.15
    gripper_goal_altitude: float = 0.05
    w_tire_linear_velocity: float = 10.0
    w_tire_angular_velocity: float = 0.30
    w_leg_proximity: float = 100.0


class SpotTire(SpotBase[SpotTireConfig]):
    """Task getting Spot to move a tire to a desired goal location."""

    def __init__(self) -> None:
        self.model_filename = str(MODEL_PATH / "xml/spot_tire_rim.xml")
        self.policy_filename = str(DATA_PATH / "policies/xinghao_policy_v1.onnx")
        super().__init__(self.model_filename, self.policy_filename)
        self.command_mask = np.arange(0, 10)

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: SpotTireConfig,
        additional_info: dict[str, Any],
    ) -> np.ndarray:
        """Reward function for the Spot box moving task."""
        batch_size = states.shape[0]

        body_height = states[..., 2]
        body_pos = states[..., 0:3]
        object_pos = states[..., 26:29]
        tire_linear_velocity = states[..., -6:-3]
        tire_angular_velocity = states[..., -3:]
        gripper_pos = sensors[..., 12:15]

        tire_goal = np.array(config.goal_position)[None, None]
        tire_to_goal = tire_goal - object_pos
        tire_to_goal_norm = np.linalg.norm(tire_to_goal, axis=-1, keepdims=-1)
        tire_to_goal_direction = tire_to_goal / (1e-2 + tire_to_goal_norm)

        gripper_goal = object_pos - config.gripper_goal_offset * tire_to_goal_direction
        gripper_goal[..., 2] = config.gripper_goal_altitude
        torso_goal = object_pos - config.torso_goal_offset * tire_to_goal_direction

        # Check if any state in the rollout has spot fallen
        spot_fallen_reward = -config.fall_penalty * (
            body_height <= config.spot_fallen_threshold
        ).any(axis=-1)
        tire_fallen_reward = -config.fall_penalty * np.abs(
            np.dot(sensors[..., 3:6], Z_AXIS) > config.tire_fallen_threshold
        ).sum(axis=-1)

        # Compute l2 distance from tire pos. to goal.
        goal_reward = -config.w_goal * np.linalg.norm(
            object_pos - config.goal_tire_pos, axis=-1
        ).mean(-1)

        # Compute l2 distance from torso pos. to tire pos.
        torso_proximity_reward = -config.w_torso_proximity * np.linalg.norm(
            body_pos - torso_goal, axis=-1
        ).mean(-1)

        # Compute l2 distance from gripper pos. to offset tire pos.
        gripper_proximity_reward = -config.w_gripper_proximity * np.linalg.norm(
            gripper_goal - gripper_pos,
            axis=-1,
        ).mean(-1)

        # Compute (thresholded) distance from legs to tire.
        leg_proximity_reward = config.w_leg_proximity * (sensors[..., -4:] - 0.1).sum(
            -1
        ).mean(-1)

        # Compute a velocity penalty to prefer small velocity commands.
        controls_reward = -config.w_controls * np.linalg.norm(controls, axis=-1).mean(
            -1
        )

        # Compute a tire velocity reward to penalize the tire rolling too much.
        tire_linear_velocity_reward = -config.w_tire_linear_velocity * np.linalg.norm(
            tire_linear_velocity, axis=-1
        ).mean(-1)
        tire_angular_velocity_reward = -config.w_tire_angular_velocity * np.linalg.norm(
            tire_angular_velocity, axis=-1
        ).mean(-1)

        assert spot_fallen_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert torso_proximity_reward.shape == (batch_size,)
        assert gripper_proximity_reward.shape == (batch_size,)
        assert leg_proximity_reward.shape == (batch_size,)
        assert controls_reward.shape == (batch_size,)
        assert tire_linear_velocity_reward.shape == (batch_size,)
        assert tire_angular_velocity_reward.shape == (batch_size,)
        assert tire_fallen_reward.shape == (batch_size,)

        reward = (
            spot_fallen_reward
            + goal_reward
            + torso_proximity_reward
            + gripper_proximity_reward
            + leg_proximity_reward
            + controls_reward
            + tire_linear_velocity_reward
            + tire_angular_velocity_reward
            + tire_fallen_reward
        )
        return reward

    def reset(self) -> None:
        """Reset function for the spot tire manipulation task ."""
        standing_pose = np.array([0, 0, 0.52])
        robot_radius = 1.0
        reset_pose = (
            np.random.rand(
                7,
            )
            - 0.5
        ) * 3.0
        reset_pose[2] = 0.275
        reset_pose[3:] = [1, 0, 0, 0]
        while np.linalg.norm(reset_pose[:3] - standing_pose) < robot_radius:
            reset_pose = (
                np.random.rand(
                    7,
                )
                - 0.5
            ) * 3.0
            reset_pose[2] = 0.275
            reset_pose[3:] = [1, 0, 0, 0]
        self.data.qpos = np.array(
            [0, 0, 0.52, 1, 0, 0, 0, *STANDING_UNSTOWED_POS, *reset_pose]
        )
        self.data.qvel = np.zeros_like(self.data.qvel)
        mujoco.mj_forward(self.model, self.data)
        self.last_policy_output = np.copy(self.initial_policy_output)
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
                -0.5 * np.ones(3),
                ARM_UNSTOWED_POS - 0.7 * np.array([1, 1.2, 1, 1, 1, 1, 0]),
            )
        )
        upper_bound = np.concatenate(
            (
                0.5 * np.ones(3),
                ARM_UNSTOWED_POS + 0.7 * np.array([1, 1.2, 1, 1, 1, 1, 0]),
            )
        )
        return np.stack([lower_bound, upper_bound], axis=-1)
