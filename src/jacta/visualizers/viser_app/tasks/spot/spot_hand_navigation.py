# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from jacta.visualizers.mujoco_helpers.utils import get_sensor_name
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


@dataclass
class SpotHandNavigationConfig(SpotBaseConfig):
    """Config for the spot box manipulation task."""

    default_command: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 0.0, *ARM_UNSTOWED_POS])
    )
    goal_position: np.ndarray = GOAL_POSITIONS().black_cross


class SpotHandNavigation(SpotBase[SpotHandNavigationConfig]):
    """Task getting Spot to navigate to a desired goal location."""

    def __init__(self) -> None:
        self.model_filename: str = str(MODEL_PATH / "xml/spot_locomotion.xml")
        self.policy_filename: str = str(DATA_PATH / "policies/xinghao_policy_v1.onnx")
        super().__init__(self.model_filename, self.policy_filename)
        self.command_mask = np.arange(0, 10)

        self.hand_sensor_adr = -1

        # Get hand sensor.
        for i in range(self.model.nsensor):
            sensor_name = get_sensor_name(self.model, i)
            if sensor_name == "trace_gripper":
                self.hand_sensor_adr = self.model.sensor_adr[i]

        assert self.hand_sensor_adr != -1, "Hand sensor not found."

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: SpotHandNavigationConfig,
        additional_info: dict[str, Any],
    ) -> np.ndarray:
        """Reward function for the Spot navigation task."""
        batch_size = states.shape[0]

        # Check if any state in the rollout has spot fallen
        body_height = states[..., 2]
        spot_fallen_reward = -config.fall_penalty * (
            body_height <= config.spot_fallen_threshold
        ).any(axis=-1)

        # DIRTY: repurposing gripper trace for reward comp.
        spot_hand_loc = sensors[..., self.hand_sensor_adr : self.hand_sensor_adr + 3]
        goal_reward = -config.w_goal * np.linalg.norm(
            spot_hand_loc - np.array(config.goal_position)[None, None], axis=-1
        ).mean(-1)

        # Compute a velocity penalty to prefer small velocity commands.
        controls_reward = -config.w_controls * np.linalg.norm(controls, axis=-1).mean(
            -1
        )

        assert spot_fallen_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert controls_reward.shape == (batch_size,)

        return spot_fallen_reward + goal_reward + controls_reward

    def reset(self) -> None:
        """Reset function for the spot navigation task ."""
        self.data.qpos = np.array([0, 0, 0.52, 1, 0, 0, 0, *STANDING_UNSTOWED_POS])
        self.data.qpos[:2] = np.random.randn(2)
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
            (-0.5 * np.ones(3), ARM_UNSTOWED_POS - 0.7 * np.array([1] * 6 + [0]))
        )
        upper_bound = np.concatenate(
            (0.5 * np.ones(3), ARM_UNSTOWED_POS + 0.7 * np.array([1] * 6 + [0]))
        )
        return np.stack([lower_bound, upper_bound], axis=-1)
