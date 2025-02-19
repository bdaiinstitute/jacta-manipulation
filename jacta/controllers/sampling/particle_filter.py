# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.


from dataclasses import dataclass
from typing import Any

import numpy as np

from jacta.plannermanipulation.controllers.sampling_base import SamplingBase, SamplingBaseConfig, make_spline
from jacta.plannermanipulation.tasks.task import Task, TaskConfig
from jacta.plannermanipulation.viser_app.gui import slider


@slider("temperature", 0.0, 100.0, 0.25)
@dataclass
class ParticleFilterConfig(SamplingBaseConfig):
    """Configuration for cross-entropy method."""

    sigma: float = 0.5
    temperature: float = 1.0


class ParticleFilter(SamplingBase):
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
        config: ParticleFilterConfig,
        reward_config: TaskConfig,
    ):
        super().__init__(task, config, reward_config)

        # Preallocate state / control buffers.
        self.all_splines = make_spline(task.data.time + self.spline_timesteps, self.controls, self.config.spline_order)

    def update_action(self, curr_state: np.ndarray, curr_time: float, additional_info: dict[str, Any]) -> None:
        """Performs rollouts + reward computation from current state."""
        assert curr_state.shape == (self.model.nq + self.model.nv,)
        assert self.config.num_rollouts > 0, "Need at least one rollout!"

        # Check if num_rollouts has changed and resize arrays accordingly.
        if len(self.models) != self.config.num_rollouts:
            self.make_models()
            self.controls: np.ndarray = np.random.default_rng().choice(self.controls, size=self.config.num_rollouts)
            self.all_splines = make_spline(curr_time + self.spline_timesteps, self.controls, self.config.spline_order)

        # Adjust time + move policy forward.
        # TODO(pculbert): move some of this logic into top-level controller.
        new_times = curr_time + self.spline_timesteps
        base_controls = self.all_splines(new_times)

        # Sample action noise (leaving one sequence noiseless).
        self.candidate_controls = base_controls + self.config.sigma * np.random.randn(
            self.config.num_rollouts, self.config.num_nodes, self.task.nu
        )

        # Clamp controls to action bounds.
        self.candidate_controls = np.clip(
            self.candidate_controls, self.task.actuator_ctrlrange[:, 0], self.task.actuator_ctrlrange[:, 1]
        )

        # Evaluate rollout controls at sim timesteps.
        candidate_splines = make_spline(new_times, self.candidate_controls, self.config.spline_order)
        rollout_controls = candidate_splines(curr_time + self.rollout_times)

        # Create lists of states / controls for rollout.
        curr_state_batch = np.tile(curr_state, (self.config.num_rollouts, 1))

        # Roll out dynamics with action sequences and set the cutoff time for each controller here
        self.task.cutoff_time = self.reward_config.cutoff_time

        # Roll out dynamics with action sequences.
        self.states, self.sensors = self.task.rollout(self.models, curr_state_batch, rollout_controls, additional_info)

        # Evalate rewards
        self.rewards = self.reward_function(self.states, self.sensors, rollout_controls, self.reward_config)

        # Compute particle filter weights.
        rewards_centered = self.rewards - self.rewards.max()
        weights = np.exp(self.config.temperature * rewards_centered)
        weights = weights / np.sum(weights, keepdims=True)

        # Sample new particles.
        self.controls = np.random.default_rng().choice(
            self.candidate_controls,
            size=self.config.num_rollouts,
            p=weights,
        )

        # Set spline to first control (arbitrarily).
        self.update_spline(new_times, self.controls[0])
        self.all_splines = make_spline(new_times, self.controls, self.config.spline_order)

        # Update traces.
        self.update_traces()

    def action(self, time: float) -> np.ndarray:
        """Current best action of policy."""
        return self.spline(time)
