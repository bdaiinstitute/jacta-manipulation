# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from jacta.visualizers.viser_app.constants import STANDING_STOWED_POS
from jacta.visualizers.viser_app.path_utils import DATA_PATH, MODEL_PATH
from jacta.visualizers.viser_app.tasks.spot_base import (
    DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME,
    GOAL_POSITIONS,
    SpotBase,
    SpotBaseConfig,
)


@dataclass
class SpotNavigationConfig(SpotBaseConfig):
    """Config for the spot box manipulation task."""

    default_command: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0.0]))
    goal_position: np.ndarray = GOAL_POSITIONS().origin


class SpotNavigation(SpotBase[SpotNavigationConfig]):
    """Task getting Spot to navigate to a desired goal location."""

    def __init__(self) -> None:
        self.model_filename = str(MODEL_PATH / "xml/spot_locomotion.xml")
        self.policy_filename = str(DATA_PATH / "policies/xinghao_policy_v1.onnx")
        super().__init__(self.model_filename, self.policy_filename)
        self.command_mask = np.arange(0, 3)

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: SpotNavigationConfig,
        additional_info: dict[str, Any],
    ) -> np.ndarray:
        """Reward function for the Spot navigation task."""
        batch_size = states.shape[0]

        body_height = states[..., 2]
        body_pos = states[..., 0:3]

        # Check if any state in the rollout has spot fallen
        spot_fallen_reward = -config.fall_penalty * (
            body_height <= config.spot_fallen_threshold
        ).any(axis=-1)

        # Compute l2 distance from torso pos. to goal.
        goal_reward = -config.w_goal * np.linalg.norm(
            body_pos - np.array(config.goal_position)[None, None], axis=-1
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
        self.data.qpos = np.array([0, 0, 0.52, 1, 0, 0, 0, *STANDING_STOWED_POS])
        self.data.qvel = np.zeros_like(self.data.qvel)
        mujoco.mj_forward(self.model, self.data)
        self.last_policy_output = np.copy(self.initial_policy_output)
        self._additional_info["last_policy_output"] = self.initial_policy_output
        self.cutoff_time = DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME

    @property
    def nu(self) -> int:
        """Number of controls for this task."""
        return 3

    @property
    def actuator_ctrlrange(self) -> np.ndarray:
        """Control bounds for this task."""
        lower_bound = -0.5 * np.ones(3)
        upper_bound = 0.5 * np.ones(3)
        return np.stack([lower_bound, upper_bound], axis=-1)
