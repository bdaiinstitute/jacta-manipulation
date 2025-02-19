# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass, field

import mujoco
import numpy as np

from jacta.plannermanipulation.common.constants import ARM_UNSTOWED_POS, STANDING_UNSTOWED_POS
from jacta.plannermanipulation.tasks.spot_base import DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME, GOAL_POSITIONS, SpotBase, SpotBaseConfig

# mocap area cross locations
RESET_OBJECT_POSE = np.array([3, 0, 0.275, 1, 0, 0, 0])
Z_AXIS = np.array([0.0, 0.0, 1.0])


@dataclass
class SpotBoxConfig(SpotBaseConfig):
    """Config for the spot box manipulation task."""

    default_command: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0.0, *ARM_UNSTOWED_POS]))
    goal_position: np.ndarray = GOAL_POSITIONS["blue_cross"]
    w_orientation: float = 5.0
    w_torso_proximity: float = 0.1
    w_gripper_proximity: float = 4.0
    orientation_threshold: float = 0.5


class SpotBox(SpotBase[SpotBoxConfig]):
    """Task getting Spot to move a box to a desired goal location."""

    def __init__(self) -> None:
        self.model_filename = "dexterity/models/xml/scenes/legacy/spot_box_lara.xml"
        self.policy_filename = "dexterity/data/policies/xinghao_policy_friday.onnx"
        super().__init__(self.model_filename, self.policy_filename)
        self.command_mask = np.arange(0, 10)

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: SpotBoxConfig,
    ) -> np.ndarray:
        """Reward function for the Spot box moving task."""
        batch_size = states.shape[0]

        # Check if any state in the rollout has spot fallen
        spot_fallen_reward = -config.fall_penalty * (states[..., 2] <= config.spot_fallen_threshold).any(axis=-1)

        # Compute l2 distance from tire pos. to goal.
        goal_reward = -config.w_goal * np.linalg.norm(
            states[..., 26:29] - np.array(config.goal_position)[None, None], axis=-1
        ).mean(-1)

        box_orientation_reward = -config.w_orientation * np.abs(
            np.dot(sensors[..., 3:6], Z_AXIS) > config.orientation_threshold
        ).sum(axis=-1)

        # Compute l2 distance from torso pos. to tire pos.
        torso_proximity_reward = config.w_torso_proximity * np.linalg.norm(
            states[..., 0:3] - states[..., 26:29], axis=-1
        ).mean(-1)

        # Compute l2 distance from torso pos. to tire pos.
        gripper_proximity_reward = -config.w_gripper_proximity * np.linalg.norm(
            sensors[..., 6:9],
            axis=-1,
        ).mean(-1)

        # Compute a velocity penalty to prefer small velocity commands.
        vel_reward = -config.w_vel * np.linalg.norm(controls, axis=-1).mean(-1)

        assert spot_fallen_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert box_orientation_reward.shape == (batch_size,)
        assert torso_proximity_reward.shape == (batch_size,)
        assert gripper_proximity_reward.shape == (batch_size,)
        assert vel_reward.shape == (batch_size,)

        return (
            spot_fallen_reward
            + goal_reward
            + box_orientation_reward
            + torso_proximity_reward
            + gripper_proximity_reward
            + vel_reward
        )

    def reset(self) -> None:
        """Reset function for the spot box manipulation task ."""
        self.data.qpos = np.array([0, 0, 0.52, 1, 0, 0, 0, *STANDING_UNSTOWED_POS, *RESET_OBJECT_POSE])
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
        lower_bound = np.concatenate((-0.7 * np.ones(3), ARM_UNSTOWED_POS - 0.7 * np.array([1] * 6 + [0])))
        upper_bound = np.concatenate((0.7 * np.ones(3), ARM_UNSTOWED_POS + 0.7 * np.array([1] * 6 + [0])))
        return np.stack([lower_bound, upper_bound], axis=-1)
