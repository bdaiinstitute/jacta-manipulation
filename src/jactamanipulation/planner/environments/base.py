# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import FloatTensor, IntTensor

from dexterity.planner.dynamics.mujoco_dynamics import MujocoPlant
from dexterity.planner.planner.action_processor import ActionProcessor
from dexterity.planner.planner.graph import sample_random_goal_states, sample_random_start_states
from dexterity.planner.planner.parameter_container import ParameterContainer
from dexterity.planner.planner.types import ActionType


@dataclass
class DexterityEnv:
    """Base environment class for dexterity tasks."""

    params: ParameterContainer

    def __post_init__(self) -> None:
        self.num_envs = self.params.num_envs
        self.all_envs = torch.arange(self.num_envs)
        self.episode_length = self.params.learner_trajectory_length
        self.setup_plant()
        self.setup_buffers()
        self.setup_action_processor()
        self.action_magnitude = self.params.action_range * self.params.action_time_step

    def setup_plant(self) -> None:
        """Set up the MujocoPlant for the environment."""
        self.plant = MujocoPlant(params=self.params)

    def setup_action_processor(self) -> None:
        """Set up the ActionProcessor for the environment."""
        self.action_processor: ActionProcessor = self.params.action_processor_class(
            params=self.params, actuated_pos=self.plant.actuated_pos
        )

    def setup_buffers(self) -> None:
        """Initialize buffers for storing environment state and trackers."""
        plant_state_size = (self.num_envs, self.plant.state_dimension)
        plant_action_size = (self.num_envs, self.plant.action_dimension)
        # initialize state placeholders
        self.start = torch.zeros(plant_state_size, dtype=torch.float32)
        self.current = torch.zeros(plant_state_size, dtype=torch.float32)
        self.goal = torch.zeros(plant_state_size, dtype=torch.float32)
        # initialize environment trackers
        self.timestep = torch.full((self.num_envs,), -1)
        self.rewards = torch.zeros(self.num_envs, dtype=torch.float32)
        self.needs_reset = torch.ones(self.num_envs, dtype=torch.bool)
        # initialize previous action placeholder
        self.previous_end_actions = torch.zeros(plant_action_size, dtype=torch.float32)

    @property
    def observation_space(self) -> SimpleNamespace:
        """Return the observation space of the environment."""
        return SimpleNamespace(
            shape=(self.plant.state_dimension * 2,),
        )

    @property
    def action_space(self) -> SimpleNamespace:
        """Return the action space of the environment."""
        return SimpleNamespace(
            low=-self.action_magnitude,
            high=self.action_magnitude,
            shape=(self.plant.action_dimension,),
        )

    def scale_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Scale the given actions to the appropriate range."""
        return actions * self.action_magnitude

    def uniform_random_action(self) -> torch.Tensor:
        """Generate uniform random actions for all environments."""
        random = torch.rand((self.num_envs, *self.action_space.shape)) * 2.0 - 1.0
        return self.scale_actions(random)

    def get_metrics(self) -> dict:
        """Calculate and return metrics for the current state of the environment."""
        current_distance = self.plant.scaled_distances_to(self.current, self.goal)
        initial_distance = self.plant.scaled_distances_to(self.start, self.goal)
        return {
            "scaled_distance_to_goal": current_distance,
            "progress": current_distance / initial_distance,
        }

    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset the environment and return the initial observations and info dictionary.

        This reset only needs to be called once at the beginning of the training loop.
        Otherwise, the environment is reset automatically when individual episodes end.
        """
        self.reset_ixs(ixs=torch.arange(self.num_envs))
        self.post_physics_step()
        return self.obs, self.info

    def reset_ixs(self, ixs: IntTensor) -> None:
        """Reset specific environments identified by their indices."""
        # sample new start and goal states
        self.reset_states(ixs)  # TODO(maks) refactor to explicity accept comprehensive reset state
        # reset episodic trackers
        self.needs_reset[ixs] = False
        self.timestep[ixs] = -1

    def update_joint_targets(self, actions: torch.Tensor) -> torch.Tensor:
        """Update joint targets based on the given actions."""
        # clamp actions
        actions = torch.clamp(actions, -self.action_magnitude, self.action_magnitude)

        # calculate actions
        _, new_end_actions, self.action_trajectories = self.action_processor(
            relative_actions=actions,
            action_type=ActionType.NON_EXPERT,
            current_states=self.current,
            previous_end_actions=self.previous_end_actions,
        )
        self.previous_end_actions = new_end_actions

    def step_sim(self) -> None:
        """Perform a simulation step using the plant dynamics."""
        self.current, _, self.state_trajectory = self.plant.dynamics(
            states=self.current,
            action_trajectories=self.action_trajectories,
        )

    def post_physics_step(self) -> None:
        """Perform post-processing after a physics step."""
        # update trackers
        self.timestep += 1
        # get metrics
        self.metrics = self.get_metrics()
        # compute rewards and termination
        self.update_terminations()
        self.update_rewards()
        self.update_info()
        # preserve last observations of reset environments
        self.preserve_reset_obs()
        # reset environments that are done
        self.process_resets()
        # process observations
        self.update_obs()

    def check_success(self) -> torch.Tensor:
        """Check if the current state meets the success criteria."""
        is_success = self.metrics["progress"] < self.params.success_progress_threshold
        return is_success

    def check_failure(self) -> torch.Tensor:
        """Check if the current state meets the failure criteria."""
        is_failure = torch.zeros_like(self.needs_reset, dtype=torch.bool)
        return is_failure

    def update_terminations(self) -> None:
        """Update termination conditions for all environments."""
        # success and failure conditions
        self.is_success = self.check_success()
        self.is_failure = self.check_failure()

        # termination
        self.termination = torch.logical_or(self.is_success, self.is_failure)

        # truncation
        self.truncation = self.timestep >= self.episode_length - 1

        # done
        self.done = torch.logical_or(self.termination, self.truncation)
        self.needs_reset[self.done] = True

    def task_rewards(self) -> torch.Tensor:
        """Calculate task-specific rewards."""
        if self.params.learner_use_sparse_reward:
            return torch.full_like(self.rewards, -1.0)
        else:
            return -self.metrics["progress"] * self.params.reward_progress_scale

    def update_rewards(self) -> None:
        """Update rewards for all environments."""
        self.rewards *= 0.0

        # task progress reward
        self.rewards += self.task_rewards()

        # success reward
        self.rewards[self.is_success] += self.params.reward_success_score

        # failure penalty
        self.rewards[self.is_failure] += self.params.reward_failure_score

        # normalize total reward for the transition
        self.rewards /= self.params.reward_maximum_magnitude

    def process_resets(self) -> None:
        """Process resets for environments that need it."""
        reset_env_ixs = torch.where(self.needs_reset)[0]
        if len(reset_env_ixs) > 0:
            self.reset_ixs(reset_env_ixs)

    def update_obs(self) -> None:
        """Update the observation tensor."""
        self.obs = torch.cat([self.current, self.goal], dim=-1)

    def preserve_reset_obs(self) -> None:
        """Preserve the last observations of environments that need to be reset.

        Used to query the Q-function for the last observation of non-terminal last steps.
        """
        if hasattr(self, "obs"):
            self.final_obs = self.obs.clone()

    @property
    def metrics_keys(self) -> list[str]:
        """Return the keys of the metrics for the environment."""
        return ["progress", "scaled_distance_to_goal"]

    def update_info(self) -> None:
        """Update the info dictionary with current environment information."""
        self.info = dict()
        self.info["done"] = self.done
        self.info["timestep"] = self.timestep.clone()
        self.info["is_success"] = self.is_success.clone()
        self.info["is_failure"] = self.is_failure.clone()
        self.info.update({key: self.metrics[key].clone() for key in self.metrics_keys})

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Perform a step in the environment given the actions."""
        # convert actions to joint targets
        self.update_joint_targets(actions)
        # run simulation
        self.step_sim()
        # post physics step
        self.post_physics_step()
        # return experience tuple
        return self.obs.clone(), self.rewards.clone(), self.termination.clone(), self.truncation.clone(), self.info

    def reset_states(self, env_ixs: IntTensor) -> None:
        """Reset the states for specific environments."""
        num_resets = len(env_ixs)

        start_states = sample_random_start_states(self.plant, self.params, num_resets)
        self.set_start_states(start_states, env_ixs)
        self.set_current_states(start_states, env_ixs)

        goal_states = sample_random_goal_states(self.plant, self.params, num_resets)
        self.set_goal_states(goal_states, env_ixs)

        # set previous actions to current joint positions
        self.set_previous_end_actions(start_states, env_ixs)

    def set_previous_end_actions(self, start_states: FloatTensor, env_ixs: IntTensor) -> None:
        """Set the previous end actions for specific environments."""
        self.previous_end_actions[env_ixs] = start_states[env_ixs[:, None], self.plant.actuated_pos]

    def set_current_states(self, states: FloatTensor, env_ixs: IntTensor) -> None:
        """Set the current states for specific environments."""
        self.current[env_ixs] = states

    def set_goal_states(self, states: FloatTensor, env_ixs: IntTensor) -> None:
        """Set the goal states for specific environments."""
        self.goal[env_ixs] = states

    def set_start_states(self, states: FloatTensor, env_ixs: IntTensor) -> None:
        """Set the start states for specific environments."""
        self.start[env_ixs] = states

    def close(self) -> None:
        """Close the environment and perform any necessary cleanup."""
