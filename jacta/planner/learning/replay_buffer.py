# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
from typing import Callable, Tuple

import torch
from torch import FloatTensor, IntTensor

from jacta.planner.dynamics.simulator_plant import SimulatorPlant
from jacta.planner.planner.parameter_container import ParameterContainer


class ReplayBuffer:
    def __init__(self, plant: SimulatorPlant, params: ParameterContainer):
        self._initialize(plant, params)

    def reset(self) -> None:
        self._initialize(self.plant, self.params)

    def _initialize(self, plant: SimulatorPlant, params: ParameterContainer) -> None:
        self.params = params
        self.plant = plant

        self.first_learner_id = (
            params.learner_trajectory_length * params.learner_evals
        )  # start id for the learner's replay buffer

        self.next_temporary_id = 0
        self.next_learner_id = self.first_learner_id
        # the largest valid id in the learning buffer, because restarting at beginning when full
        self.max_learner_id = self.next_learner_id - 1

        # Create all data containers
        self.max_nodes = (params.learner_evals + params.learner_max_trajectories) * params.learner_trajectory_length

        self.states = torch.zeros((self.max_nodes, plant.state_dimension))
        self.start_actions = torch.zeros((self.max_nodes, plant.action_dimension))
        self.end_actions = torch.zeros((self.max_nodes, plant.action_dimension))
        self.relative_actions = torch.zeros(
            (self.max_nodes, plant.action_dimension)
        )  # without clipping to absolute bounds
        self.learning_goals = torch.zeros((self.max_nodes, plant.state_dimension))
        self.parents = torch.zeros(self.max_nodes, dtype=torch.int64)
        self.root_ids = torch.arange(self.max_nodes, dtype=torch.int64)

    def reset_next_temporary_id(self) -> None:
        self.next_temporary_id = 0

    def add_nodes(
        self,
        root_ids: IntTensor,
        parent_ids: IntTensor,
        states: FloatTensor,
        start_actions: FloatTensor,
        end_actions: FloatTensor,
        relative_actions: FloatTensor,
        temporary: bool = False,
        sub_goal_state: FloatTensor = None,
    ) -> int:
        if temporary:
            ids = torch.arange(self.next_temporary_id, self.next_temporary_id + len(parent_ids))
            self.next_temporary_id = ids[-1] + 1
        else:
            ids = torch.arange(self.next_learner_id, self.next_learner_id + len(parent_ids))
            if ids[-1] >= self.max_nodes:  # end of buffer reached, start from beginning
                ids = torch.arange(self.first_learner_id, self.first_learner_id + len(parent_ids))
                parent_ids = ids - 1
                parent_ids[0] += 1  # first node is its own parent

            self.next_learner_id = ids[-1] + 1
            self.max_learner_id = max(self.max_learner_id, self.next_learner_id - 1)
            if self.next_learner_id >= self.max_nodes:
                self.next_learner_id = self.first_learner_id

        self.root_ids[ids] = root_ids
        self.parents[ids] = parent_ids
        self.states[ids] = states
        self.start_actions[ids] = start_actions
        self.end_actions[ids] = end_actions
        self.relative_actions[ids] = relative_actions
        self.learning_goals[ids] = sub_goal_state

        return ids

    def sampling(
        self,
        batch_size: int,
        her_probability: float,
        reward_function: Callable,
    ) -> Tuple[
        FloatTensor,
        FloatTensor,
        FloatTensor,
        FloatTensor,
        FloatTensor,
        FloatTensor,
        FloatTensor,
        FloatTensor,
    ]:
        """Sample a batch at random with HER goal resampling from replay experience.

        Args:
            batch_size: Batch size.
            her_probability: Probability of resampling a goal with an achieved goal.
            reward_function: Reward function of the environment to recalculate the rewards.

        Returns:
            A tuple of the sampled state, action, reward, next_state, goal batch.

        Raises:
            Assertion error: Dimension check on states failed.
        """
        params = self.params

        T = params.learner_trajectory_length - 1
        action_normalization_scaling = params.action_range * params.action_time_step
        learning_ids = torch.arange(self.first_learner_id, self.max_learner_id + 1)
        learning_ids = learning_ids.reshape((-1, T + 1))
        num_episodes = learning_ids.shape[0]

        episode_indices = torch.randint(0, num_episodes, (batch_size,))
        timestep_indices = torch.randint(0, T, (batch_size,))

        her_episode_indices = torch.where(torch.rand(batch_size) < her_probability)[0]  # relabling indices

        timestep_offset = (torch.rand(batch_size) * (T - timestep_indices)).to(torch.int)

        her_timestep_indices = (timestep_indices + 1 + timestep_offset)[her_episode_indices]

        current_node_ids = learning_ids[episode_indices, timestep_indices]
        next_node_ids = learning_ids[episode_indices, timestep_indices + 1]
        her_node_ids = learning_ids[episode_indices[her_episode_indices], her_timestep_indices]

        states = self.states[current_node_ids].clone()
        # actions from current node are stored in next node
        actions = self.relative_actions[next_node_ids].clone() / action_normalization_scaling
        goals = self.learning_goals[current_node_ids].clone()
        goals[her_episode_indices] = self.states[her_node_ids].clone()
        rewards, _ = reward_function(self, next_node_ids, goals)
        rewards = torch.unsqueeze(rewards, dim=1)
        next_states = self.states[next_node_ids].clone()

        # the ids are returned for testing
        return states, actions, rewards, next_states, goals, current_node_ids, next_node_ids, her_node_ids
