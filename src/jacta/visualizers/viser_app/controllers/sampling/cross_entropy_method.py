# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.


from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import interp1d

from jacta.visualizers.viser_app.controllers.sampling_base import (
    SamplingBase,
    SamplingBaseConfig,
    make_spline,
)
from jacta.visualizers.viser_app.tasks.task import Task, TaskConfig

#### BOX HARDWARE
# @dataclass
# class CrossEntropyConfig(SamplingBaseConfig):
#     """Configuration for cross-entropy method."""

#     sigma_min: float = 0.3
#     sigma_max: float = 1.0
#     num_elites: int = 2
#     horizon: float = 3.8
#     num_rollouts: int = 48
#     noise_ramp: float = 10.
#     use_noise_ramp: bool = True


# ## YELLOW CHAIR HARDWARE
# @dataclass
# class CrossEntropyConfig(SamplingBaseConfig):
#     """Configuration for cross-entropy method."""

#     sigma_min: float = 0.1
#     sigma_max: float = 1.0
#     num_elites: int = 2
#     horizon: float = 3.8
#     num_rollouts: int = 48
#     noise_ramp: float = 2.5
#     use_noise_ramp: bool = True


@dataclass
class CrossEntropyConfig(SamplingBaseConfig):
    """Configuration for cross-entropy method."""

    sigma_min: float = 0.1
    sigma_max: float = 1.0
    num_elites: int = 2
    horizon: float = 2.8
    num_rollouts: int = 32
    noise_ramp: float = 2.5
    use_noise_ramp: bool = True


class CrossEntropyMethod(SamplingBase):
    """The cross-entropy method.

    Args:
        config: configuration object with hyperparameters for planner.
        model: mujoco model of system being controlled.
        data: current configuration data for mujoco model.
        reward_func: function mapping batches of states/controls to batches of rewards.
    """

    def __init__(
        self,
        task: Task,
        config: CrossEntropyConfig,
        reward_config: TaskConfig,
    ):
        super().__init__(task, config, reward_config)

        # Compute initial sigma value.
        self.sigma = (
            (self.config.sigma_min + self.config.sigma_max) / 2
        ) * np.ones_like(self.controls)

    def update_action(
        self, curr_state: np.ndarray, curr_time: float, additional_info: dict[str, Any]
    ) -> None:
        """Performs rollouts + reward computation from current state."""
        assert curr_state.shape == (self.model.nq + self.model.nv,)
        assert self.config.num_rollouts > 0, "Need at least one rollout!"
        assert (
            self.config.num_elites <= self.config.num_rollouts
        ), "Elite fraction must be <= 100% of number of rollouts!"

        # Check if num_rollouts has changed and resize arrays accordingly.
        if self.states.shape[:2] != (self.config.num_rollouts, self.num_timesteps):
            self.resize_data()

        # Adjust time + move policy forward.
        # TODO(pculbert): move some of this logic into top-level controller.
        new_times = curr_time + self.spline_timesteps
        base_controls = self.spline(new_times)[None]

        if len(self.sigma) != self.config.num_nodes:
            # Number of nodes has changed -- resize sigma via interpolation.
            self.sigma = interp1d(
                self.spline.x,
                self.sigma,
                axis=0,
                fill_value="extrapolate",
                kind=self.config.spline_order,
            )(new_times)

        # Sample action noise (leaving one sequence noiseless). Optional ramp up in terms of noise
        if self.config.use_noise_ramp:
            # Ramp shape needs to be extended to (num_nodes x 1)
            ramp = np.linspace(
                self.config.noise_ramp / self.config.num_nodes,
                self.config.noise_ramp,
                self.config.num_nodes,
                endpoint=True,
            )[:, None]
            self.sigma = np.clip(
                self.sigma * ramp, self.config.sigma_min, self.config.sigma_max
            )
            noised_controls = base_controls + self.sigma[None] * np.random.randn(
                self.config.num_rollouts - 1, self.config.num_nodes, self.task.nu
            )
        else:
            noised_controls = base_controls + self.sigma[None] * np.random.randn(
                self.config.num_rollouts - 1, self.config.num_nodes, self.task.nu
            )

        self.candidate_controls = np.concatenate([base_controls, noised_controls])

        # Clamp controls to action bounds.
        self.candidate_controls = np.clip(
            self.candidate_controls,
            self.task.actuator_ctrlrange[:, 0],
            self.task.actuator_ctrlrange[:, 1],
        )

        # Evaluate rollout controls at sim timesteps.
        candidate_splines = make_spline(
            new_times, self.candidate_controls, self.config.spline_order
        )
        self.rollout_controls = candidate_splines(curr_time + self.rollout_times)

        # Create lists of states / controls for rollout.
        curr_state_batch = np.tile(curr_state, (self.config.num_rollouts, 1))

        # Roll out dynamics with action sequences and set the cutoff time for each controller here
        self.task.cutoff_time = self.reward_config.cutoff_time

        # Roll out dynamics with action sequences.
        self.task.rollout(
            self.models,
            curr_state_batch,
            self.rollout_controls,
            additional_info,
            self.states,
            self.sensors,
        )

        # Evalate rewards
        self.rewards = self.reward_function(
            self.states,
            self.sensors,
            self.rollout_controls,
            self.reward_config,
            additional_info,
        )
        # Compute elite indices by reverse-sorting rewards.
        self.elite_inds = np.flip(np.argsort(self.rewards))[: self.config.num_elites]

        # Store new mean/sigma for elite controls.
        self.controls: np.ndarray = self.candidate_controls[self.elite_inds].mean(0)
        self.sigma = np.clip(
            np.sqrt(self.candidate_controls[self.elite_inds].var(0)),
            self.config.sigma_min,
            self.config.sigma_max,
        )
        self.update_spline(new_times, self.controls)

        # Update traces.
        self.update_traces()

    def action(self, time: float) -> np.ndarray:
        """Current best action of policy."""
        return self.spline(time)
