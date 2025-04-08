# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.


from dataclasses import dataclass
from typing import Any

import numpy as np

from jacta.visualizers.viser_app.controllers.sampling_base import (
    SamplingBase,
    SamplingBaseConfig,
    make_spline,
)
from jacta.visualizers.viser_app.tasks.task import Task, TaskConfig


@dataclass
class PredictiveSamplingConfig(SamplingBaseConfig):
    """Configuration for predictive sampling."""

    sigma: float = 0.05
    noise_ramp: float = 1.0


class PredictiveSampling(SamplingBase):
    """Predictive sampling planner.

    Args:
        config: configuration object with hyperparameters for planner.
        model: mujoco model of system being controlled.
        data: current configuration data for mujoco model.
        reward_func: function mapping batches of states/controls to batches of rewards.
    """

    def __init__(
        self,
        task: Task,
        config: PredictiveSamplingConfig,
        reward_config: TaskConfig,
    ):
        super().__init__(task, config, reward_config)

        # Create variable for best action.
        self.best_action = 0

    def update_action(
        self, curr_state: np.ndarray, curr_time: float, additional_info: dict[str, Any]
    ) -> None:
        """Performs rollouts + reward computation from current state."""
        assert curr_state.shape == (self.model.nq + self.model.nv,)
        assert self.config.num_rollouts > 0, "Need at least one rollout!"

        # Check if num_rollouts has changed and resize arrays accordingly.
        if self.states.shape[:2] != (self.config.num_rollouts, self.num_timesteps):
            self.resize_data()

        # Adjust time + move policy forward.
        new_times = curr_time + self.spline_timesteps
        base_controls = self.spline(new_times)[None]

        # Sample action noise (leaving one sequence noiseless), with optional ramp up
        if self.config.use_noise_ramp:
            ramp = (
                self.config.noise_ramp
                * np.linspace(
                    1 / self.config.num_nodes, 1, self.config.num_nodes, endpoint=True
                )[:, None]
            )
            noised_controls = (
                base_controls
                + self.config.sigma
                * ramp
                * np.random.randn(
                    self.config.num_rollouts - 1, self.config.num_nodes, self.task.nu
                )
            )
        else:
            noised_controls = base_controls + self.config.sigma * np.random.randn(
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

        # Update max-reward index.
        self.best_action = self.rewards.argmax()

        # Update controls and spline.
        self.controls = self.candidate_controls[self.best_action]
        self.update_spline(new_times, self.controls)

        # Update traces.
        self.update_traces()

    def action(self, time: float) -> np.ndarray:
        """Current best action of policy."""
        return self.spline(time)
