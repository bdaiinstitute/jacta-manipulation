# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.


import logging
import time
from dataclasses import dataclass
from typing import Any

import cma
import numpy as np

from jacta.visualizers.viser_app.controllers.sampling_base import (
    SamplingBase,
    SamplingBaseConfig,
    make_spline,
)
from jacta.visualizers.viser_app.tasks.task import Task, TaskConfig


@dataclass
class CMAESConfig(SamplingBaseConfig):
    """Configuration for CMAES sampling."""

    sigma_init: float = 0.05
    max_iter: int = 20


class CMAES(SamplingBase):
    """CMAES planner.

    Args:
        config: configuration object with hyperparameters for planner.
        model: mujoco model of system being controlled.
        data: current configuration data for mujoco model.
        reward_func: function mapping batches of states/controls to batches of rewards.
    """

    def __init__(
        self,
        task: Task,
        config: CMAESConfig,
        reward_config: TaskConfig,
    ):
        super().__init__(task, config, reward_config)

    def update_action(
        self, curr_state: np.ndarray, curr_time: float, additional_info: dict[str, Any]
    ) -> None:
        """Performs rollouts + reward computation from current state."""
        t0 = time.time()
        assert curr_state.shape == (self.model.nq + self.model.nv,)
        assert self.config.num_rollouts > 0, "Need at least one rollout!"

        # Check if num_rollouts has changed and resize arrays accordingly.
        if self.states.shape[:2] != (self.config.num_rollouts, self.num_timesteps):
            self.resize_data()

        # Adjust time + move policy forward.
        new_times = curr_time + self.spline_timesteps
        base_controls = self.spline(new_times)[None]
        best_controls = base_controls
        # init CMA-ES
        noise_init = np.zeros((self.config.num_nodes * self.task.nu))
        cmaes = cma.CMAEvolutionStrategy(
            noise_init,
            self.config.sigma_init,
            {
                "CMA_diagonal": False,
                "verbose": -1,
                "CMA_active": False,
                "popsize": self.config.num_rollouts - 1,
                "tolfun": 1e-3,
            },
        )
        best_reward = -np.inf
        i = 0
        while not cmaes.stop() and i < self.config.max_iter:
            noise = np.array(cmaes.ask())
            noise = noise.reshape(
                self.config.num_rollouts - 1, self.config.num_nodes, self.task.nu
            )
            controls_samples = base_controls + noise

            # Sample action noise (leaving one sequence noiseless).
            candidate_controls = np.concatenate([base_controls, controls_samples])

            # Clamp controls to action bounds.
            candidate_controls = np.clip(
                candidate_controls,
                self.task.actuator_ctrlrange[:, 0],
                self.task.actuator_ctrlrange[:, 1],
            )

            # Evaluate rollout controls at sim timesteps.
            candidate_splines = make_spline(
                new_times, candidate_controls, self.config.spline_order
            )
            self.rollout_controls = candidate_splines(curr_time + self.rollout_times)

            # Create lists of states / controls for rollout.
            curr_state_batch = np.tile(curr_state, (self.config.num_rollouts, 1))

            # Roll out dynamics with action sequences and set the cutoff time for each controller here
            self.task.cutoff_time = self.reward_config.cutoff_time
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
            cmaes.tell(noise, -self.rewards[1:])  # cmaes works with negative rewards
            if np.max(self.rewards) > best_reward:
                best_reward = np.max(self.rewards[1:])
                best_controls = candidate_controls[np.argmax(self.rewards)]
            i += 1
        # Update controls and spline.
        self.update_spline(new_times, best_controls)

        # Update traces.
        self.update_traces()

        loop_time = time.time() - t0
        if loop_time > 1 / self.config.control_freq:
            logging.warning(
                f"CMAES loop took {loop_time:.3f} seconds, longer than control frequency."
            )

    def action(self, time: float) -> np.ndarray:
        """Current best action of policy."""
        return self.spline(time)
