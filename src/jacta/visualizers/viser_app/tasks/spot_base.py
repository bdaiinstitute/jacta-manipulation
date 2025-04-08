# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar

import mujoco
import numpy as np
from scipy.interpolate import interp1d

from jacta.visualizers.viser_app.constants import ARM_STOWED_POS
from jacta.visualizers.viser_app.gui import slider
from jacta.visualizers.viser_app.tasks.mujoco_task import MujocoTask
from jacta.visualizers.viser_app.tasks.task import TaskConfig
from mujoco_extensions.policy_rollout import (
    System,
    create_systems_vector,
    threaded_rollout,
)


@dataclass
class GOAL_POSITIONS:
    """Goal positions of Spot."""

    origin: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0.0]))
    blue_cross: np.ndarray = field(default_factory=lambda: np.array([2.77, 0.71, 0.3]))
    black_cross: np.ndarray = field(
        default_factory=lambda: np.array([1.63, -0.53, 0.31])
    )


DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME = 0.25  # seconds


@slider("w_goal", 50.0, 200.0)
@slider("w_controls", 0.0, 10.0)
@dataclass
class SpotBaseConfig(TaskConfig):
    """Base config for spot tasks."""

    default_command: np.ndarray = field(
        default_factory=lambda: np.array(
            [0, 0, 0] + list(ARM_STOWED_POS) + [0] * 12 + [0, 0, 0.55]
        )
    )
    fall_penalty: float = 2500.0
    spot_fallen_threshold = 0.35  # Torso height where Spot is considered "fallen"
    w_goal: float = 60.0
    w_proximity: float = 2.0
    w_controls: float = 0.0
    cutoff_time: float = DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME


ConfigT = TypeVar("ConfigT", bound=SpotBaseConfig)


class SpotBase(MujocoTask[ConfigT]):
    """Base task for spot locomotion.

    This is an 'abstract' class that should not be instantiated.
    """

    def __init__(self, model_filepath: str, policy_filepath: str) -> None:
        self.model_filepath = model_filepath
        self.policy_filepath = policy_filepath
        super().__init__(self.model_filepath)
        self.system = create_systems_vector(
            self.model_filepath, self.policy_filepath, num_systems=1
        )

        self.physics_substeps: int = 2
        self.default_policy_command = np.array(
            [0, 0, 0] + list(ARM_STOWED_POS) + [0] * 12 + [0, 0, 0.55]
        )
        self.command_mask = slice(0, 10)
        self.initial_policy_output = np.zeros((12,))
        self.last_policy_output = self.initial_policy_output
        self._additional_info["last_policy_output"] = self.initial_policy_output
        self.cutoff_time = DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME

    def sim_step(self, controls: Optional[interp1d]) -> None:
        """Spot policy physics evaluation. Evaluates policy and steps simulation forward.

        Args:
            controls: either a spline to be interpolated or None, which calls a default command
        """
        # Read current action from spline.
        if controls is None:
            current_spot_command = self.default_idle_command
        else:
            current_spot_command = controls(self.data.time)

        current_full_command = np.copy(self.default_policy_command)
        current_full_command[self.command_mask] = current_spot_command

        current_state = np.concatenate([self.data.qpos, self.data.qvel])

        next_state, _, next_policy_output = threaded_rollout(
            self.system,
            current_state[None],
            current_full_command[None, None],
            self.last_policy_output[None],
            1,
            self.physics_substeps,
        )

        assert len(next_state) == 1
        assert next_state[0].shape == (2, self.model.nq + self.model.nv)

        self.data.qpos[:] = next_state[0][-1][: self.model.nq]
        self.data.qvel[:] = next_state[0][-1][self.model.nq :]
        mujoco.mj_forward(self.model, self.data)

        self.last_policy_output = next_policy_output[0]
        self._additional_info["last_policy_output"] = next_policy_output[0]
        # Advance time since we don't use mj_step
        self.data.time += self.model.opt.timestep * self.physics_substeps

    def rollout(
        self,
        models: list[System],
        states: np.ndarray,
        controls: np.ndarray,
        additional_info: dict[str, Any],
        output_states: np.ndarray,
        output_sensors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Threaded rollout for spot RL system."""
        # optimized_commands = np.stack(control_batch)
        num_threads, num_timesteps, _ = controls.shape
        full_commands = np.tile(
            self.default_policy_command, (num_threads, num_timesteps, 1)
        )
        full_commands[..., self.command_mask] = controls

        policy_outputs = np.broadcast_to(
            additional_info["last_policy_output"],
            (num_threads, self.initial_policy_output.shape[-1]),
        )

        # TODO(pculbert): convert to in-place rollout.
        states, sensors, _ = threaded_rollout(
            models,
            states,
            full_commands,
            policy_outputs,
            states.shape[0],
            self.physics_substeps,
            self.cutoff_time,
        )

        output_states[:] = states
        output_sensors[:] = sensors

    def make_models(self, num_models: int) -> list[System]:
        """Allocates systems vector to be used for threaded rollout."""
        return create_systems_vector(
            self.model_filepath, self.policy_filepath, num_systems=num_models
        )

    @property
    def dt(self) -> float:
        """Effective timestep for this task -- each step is physics_substeps * the mujoco dt."""
        return self.model.opt.timestep * self.physics_substeps

    @property
    def default_idle_command(self) -> np.ndarray:
        """Default idling command. Must be defined."""
        return self.default_policy_command[self.command_mask]
