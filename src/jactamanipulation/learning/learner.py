# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

# Code modified from https://github.com/amacati/dextgen, MIT license
"""``Learner`` module encapsulating the Deep Deterministic Policy Gradient (DDPG) algorithm.

:class:`.Learner` initializes the actor, critic, and normalizers and takes care of checkpoints
during training as well as network loading if starting from pre-trained networks.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from tensordict import TensorDict
from torch import FloatTensor

from dexterity.learning.networks import Actor, Critic
from dexterity.learning.normalizer import Normalizer
from dexterity.learning.replay_buffer import ReplayBuffer
from dexterity.planner.dynamics.mujoco_dynamics import MujocoPlant
from dexterity.planner.planner.data_collection import find_latest_model_path, load_model, save_model
from dexterity.planner.planner.graph import Graph, sample_random_goal_states, sample_random_start_states
from dexterity.planner.planner.parameter_container import ParameterContainer
from dexterity.planner.planner.types import ActionType


@dataclass
class Learner:
    """Deep Deterministic Policy Gradient algorithm class"""

    plant: MujocoPlant
    graph: Graph
    replay_buffer: ReplayBuffer
    params: ParameterContainer
    save_local: bool = True  # TODO move to ParameterContainer config?
    save_cloud: bool = False
    load_local: bool = False
    load_cloud: bool = False
    verbose: bool = True

    def __post_init__(self) -> None:
        """Constructs a Learner object."""
        self.reset()

    def reset(self) -> None:
        """Reset the Learner for additional runs."""
        self.replay_buffer.reset()
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the Learner."""
        self.action_processor = self.params.action_processor_class(self.params, self.plant.actuated_pos)
        size_s = self.plant.state_dimension
        size_a = self.plant.action_dimension
        self.actor = Actor(size_s * 2, size_a)
        self.critic = Critic(size_s * 2, size_a)
        self.state_norm = Normalizer(self.plant.state_dimension)
        self.planner_experience_share = self.params.learner_max_planner_experience_share
        # the relative distance required to complete the task
        self.final_success_distance = self.params.learner_final_success_distance
        self.best_relative_distance = torch.inf
        self.inject_demonstration = True
        self.injections = 0

        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        task = self.params.model_filename[:-4]

        self.base_local_path = Path("dexterity/examples/learning/models") / task
        self.local_path = self.base_local_path / f"{now}/"
        self.traj_path = Path("dexterity/examples/learning/trajectories") / task / f"{now}/"
        self.base_cloud_path = Path("dexterity") / task / "models"
        self.cloud_path = self.base_cloud_path / f"{now}/"

        if self.save_local:
            print("Local models uploaded to {}.".format(self.local_path))
        if self.save_cloud:
            print("Cloud models uploaded to {}.".format(self.cloud_path))

        if self.load_local:
            assert not self.load_cloud
            model_path = find_latest_model_path(self.base_local_path)
            self.load_models(model_path, True)
        elif self.load_cloud:
            model_path = find_latest_model_path(self.base_cloud_path)
            self.load_models(model_path, False)

    def actor_actions(self, states: FloatTensor, goal_state: FloatTensor) -> FloatTensor:
        """Uses actor network to sample actions given `node_ids` to index into the replay buffer."""
        obs = torch.cat((states, goal_state), dim=1)
        with torch.no_grad():
            actions = self.actor.select_action(self.state_norm, obs)
        return actions * self.params.action_range * self.params.action_time_step

    def relative_distances_to(
        self,
        start_states: FloatTensor,
        current_states: FloatTensor,
        goal_states: FloatTensor,
    ) -> FloatTensor:
        """Scaled distance from states in a given container to `goal_states`."""
        node_distances = self.plant.scaled_distances_to(current_states, goal_states)
        root_distance = self.plant.scaled_distances_to(start_states, goal_states)
        return node_distances / root_distance

    def reward_function(
        self,
        start_states: FloatTensor,
        current_states: FloatTensor,
        goal_states: FloatTensor,
    ) -> tuple[FloatTensor, FloatTensor]:
        """Learner reward function."""
        relative_distances = self.relative_distances_to(start_states, current_states, goal_states)
        if self.params.learner_use_sparse_reward:
            reward = -(relative_distances > self.final_success_distance).float()
        else:  # clipped relative distance as reward
            reward = -torch.clamp(relative_distances, max=1.0)
        return reward, relative_distances

    def update_norm(self, states: FloatTensor, goals: FloatTensor) -> None:
        """Update the normalizers with the current episode of play experience.

        Samples the trajectory instead of taking every experience to create a goal distribution that
        is equal to what the networks encouter.
        """
        self.state_norm.update(states)
        self.state_norm.update(goals)

    def policy_rollout(self) -> TensorDict:
        """Rollout trained policy using randomly sampled start and goal states and add to the replay buffer."""
        params = self.params
        plant = self.plant
        sub_goal_state = sample_random_goal_states(plant, params, 1)

        start_states = sample_random_start_states(plant, params, 1)
        states = start_states.clone()
        previous_end_actions = states[:, self.plant.actuated_pos]

        rollout_data = []

        for _ in range(params.learner_trajectory_length - 1):
            relative_actions = self.actor_actions(states, sub_goal_state)
            _, sampled_end_actions, start_end_sub_actions = self.action_processor(
                relative_actions=relative_actions,
                action_type=ActionType.EXPERT,
                current_states=states,
                previous_end_actions=previous_end_actions,
            )
            next_states, *_ = plant.dynamics(states, start_end_sub_actions)

            experience_data = TensorDict(
                {
                    "states": states,
                    "relative_actions": relative_actions,
                    "start_states": start_states,
                    "goal_states": sub_goal_state,
                    "next_states": next_states,
                },
                batch_size=[1],
            )
            rollout_data.append(experience_data)

            states = next_states
            previous_end_actions = sampled_end_actions

        rollout_data = torch.cat(rollout_data, dim=0)

        return rollout_data

    def graph_rollout(self) -> TensorDict:
        """Trace trajectory in graph from randomly sampled start to goal states and add to the replay buffer."""
        params = self.params
        plant = self.plant
        graph = self.graph

        graph.change_sub_goal_states(sample_random_goal_states(plant, params))

        best_id = graph.get_best_id(reward_based=False)
        path_ids = graph.shortest_path_to(best_id)
        current_ids = path_ids[:-1]
        next_ids = path_ids[1:]
        path_length = len(current_ids)
        states = graph.states[current_ids]
        start_states = states[0:1].repeat(path_length, 1)
        goal_states = graph.sub_goal_states.repeat(path_length, 1)
        next_states = graph.states[next_ids]
        relative_actions = graph.relative_actions[next_ids]
        rollout_data = TensorDict(
            {
                "states": states,
                "relative_actions": relative_actions,
                "start_states": start_states,
                "goal_states": goal_states,
                "next_states": next_states,
            },
            batch_size=[path_length],
        )

        return rollout_data

    def set_demonstration_injection(self) -> None:
        """Sets demonstration injection"""
        if self.params.learner_stop_converged_injection and self.final_success_distance > 0.5:  # TODO make a param
            self.inject_demonstration = False
        else:
            # x% all the time
            self.inject_demonstration = torch.rand(1).item() < self.planner_experience_share

    def train(self, num_epochs: int = 50) -> None:
        """Train a policy to solve the environment with DDPG."""
        params = self.params
        replay_buffer = self.replay_buffer
        actor = self.actor
        critic = self.critic

        if self.verbose:
            print(f"training for {num_epochs} epochs (seed: {params.seed})")
            print("epoch | succ. rate | rel. dist. | demo share | time")

        final_success_rate = 0.0

        for epoch in range(num_epochs):
            epoch_start = time.perf_counter()
            demo_share = 0.0

            actor.train()
            for _ in range(params.learner_cycles):
                for _ in range(params.learner_rollouts):
                    self.set_demonstration_injection()
                    if self.inject_demonstration:
                        rollout_data = self.graph_rollout()
                        demo_share += 1
                    if not self.inject_demonstration or rollout_data.batch_size[0] == 0:
                        rollout_data = self.policy_rollout()
                    replay_buffer.extend(rollout_data)
                    self.update_norm(rollout_data["states"], rollout_data["goal_states"])
                for _ in range(params.learner_batches):
                    batch_sample = replay_buffer.sample()
                    self.train_actor_critic(batch_sample)
                actor.update_target()
                critic.update_target()
            actor.eval()

            demo_share /= params.learner_cycles * params.learner_rollouts
            epoch_end = time.perf_counter()
            final_success_rate, relative_distance = self.eval_agent()
            epoch_time = epoch_end - epoch_start

            if self.verbose:
                print(
                    f"{epoch:5} | {final_success_rate:10.3f} | "
                    f" {relative_distance:10.3f} | {demo_share:10.3f} | {epoch_time:.3f}s"
                )

            if relative_distance < self.best_relative_distance:
                self.best_relative_distance = relative_distance
                if self.save_local:
                    self.save_models(True)
                if self.save_cloud:
                    self.save_models(False)

            if final_success_rate >= params.learner_early_stop:
                return

    def preprocess_sample(
        self, data: TensorDict
    ) -> tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor]:
        """Preprocess sample

        Args:
            data (TensorDict): Tensor to preprocess

        Returns:
            tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor]:
                observation, next observation, actions, rewards, terminated
        """
        params = self.params

        states = data["states"]
        next_states = data["next_states"]
        goals = data["goal_states"]
        starts = data["start_states"]
        actions = data["relative_actions"] / (params.action_range * params.action_time_step)
        rewards, relative_distances = self.reward_function(starts, next_states, goals)
        rewards = rewards.unsqueeze(1)
        terminated = (relative_distances < params.learner_final_success_distance).unsqueeze(1).float()

        obs = torch.cat([states, goals], dim=1)
        obs_next = torch.cat([next_states, goals], dim=1)
        obs = self.state_norm.normalize_obs(obs)
        obs_next = self.state_norm.normalize_obs(obs_next)
        return obs, obs_next, actions, rewards, terminated

    def train_actor_critic(
        self,
        data: TensorDict,
        discount_factor: float = 0.98,
    ) -> None:
        """Train the agent and critic network with experience sampled from the replay buffer."""
        actor = self.actor
        critic = self.critic
        params = self.params

        obs, obs_next, actions, rewards, terminated = self.preprocess_sample(data)
        with torch.no_grad():
            next_actions = actor.target(obs_next)
            next_q = critic.target(obs_next, next_actions)
            assert next_q.shape == rewards.shape
            value = rewards + (1 - terminated) * discount_factor * next_q
            # Clip to minimum reward possible, geometric sum over the finite horizon with discount_factor and -1 rewards
            trajectory_length = params.learner_trajectory_length
            worst_case_sum = -1 * (1 - discount_factor**trajectory_length) / (1 - discount_factor)
            value = torch.clamp(value, min=worst_case_sum, max=0.0)

        q = critic(obs, actions)
        critic_loss = (value - q).pow(2).mean()

        actions = actor(obs)
        next_q = critic(obs, actions)
        actor_loss = -next_q.mean()
        actor.backward_step(actor_loss)
        critic.backward_step(critic_loss)

    def eval_agent(self) -> tuple[float, float]:
        """Evaluate the current agent performance on the task.

        Runs `learner_evals` times and averages the success rate.
        """
        params = self.params
        learner_evals = params.learner_evals
        success_count = 0
        progress_sum = 0.0
        for _ in range(learner_evals):
            rollout_data = self.policy_rollout()
            _, progress = self.reward_function(
                start_states=rollout_data["start_states"][0],  # TODO: num_envs > 1
                current_states=rollout_data["next_states"][-1],
                goal_states=rollout_data["goal_states"][0],
            )
            is_success = (progress < self.final_success_distance).sum()
            success_count += is_success
            progress_sum += progress.mean()

        return success_count / learner_evals, progress_sum / learner_evals

    def save_models(self, is_local: bool) -> None:
        """Save the actor and critic networks and the normalizers.

        Saves are located under `/models/<model_filename>/`.
        """
        path = self.local_path if is_local else self.cloud_path
        save_model(self.actor, path / "actor.pt", is_local)
        save_model(self.critic, path / "critic.pt", is_local)
        save_model(self.state_norm, path / "state_norm.pt", is_local)

    def load_models(self, path: Path, is_local: bool) -> None:
        """Load the actor and critic networks and the normalizers."""
        load_model(self.actor, path / "actor.pt", is_local)
        load_model(self.critic, path / "critic.pt", is_local)
        load_model(self.state_norm, path / "state_norm.pt", is_local)
