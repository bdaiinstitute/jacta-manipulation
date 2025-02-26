# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

# Code modified from https://github.com/amacati/dextgen, MIT license
"""``Learner`` module encapsulating the Deep Deterministic Policy Gradient (DDPG) algorithm.

:class:`.Learner` initializes the actor, critic, and normalizers and takes care of checkpoints
during training as well as network loading if starting from pre-trained networks.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torch
from torch import FloatTensor, IntTensor, tensor

from jacta.learning.networks import (
    Actor,
    Critic,
    train_actor_critic,
    train_actor_imitation,
    train_critic_imitation,
)
from jacta.learning.normalizer import Normalizer
from jacta.learning.replay_buffer import ReplayBuffer
from jacta.planner.core.action_sampler import ActionSampler
from jacta.planner.core.clipping_method import clip_actions
from jacta.planner.core.data_collection import (
    find_latest_model_path,
    load_model,
    save_model,
)
from jacta.planner.core.graph import (
    Graph,
    sample_random_goal_states,
    sample_random_start_states,
)
from jacta.planner.core.parameter_container import ParameterContainer
from jacta.planner.core.types import ActionMode
from jacta.planner.dynamics.simulator_plant import SimulatorPlant, scaled_distances_to


class Learner:
    """Deep Deterministic Policy Gradient algorithm class.

    Uses a state/goal normalizer and the HER sampling method to solve sparse reward environments.
    """

    def __init__(
        self,
        plant: SimulatorPlant,
        graph: Graph,
        replay_buffer: ReplayBuffer,
        params: ParameterContainer,
        save_local: bool = True,
        load_local: bool = False,
        verbose: bool = True,
    ) -> None:
        self._initialize(
            plant,
            graph,
            replay_buffer,
            params,
            save_local=save_local,
            load_local=load_local,
            verbose=verbose,
        )

    def reset(self) -> None:
        self.replay_buffer.reset()

        self._initialize(
            self.plant,
            self.graph,
            self.replay_buffer,
            self.params,
            save_local=self.save_local,
            load_local=self.load_local,
            verbose=self.verbose,
        )

    def _initialize(
        self,
        plant: SimulatorPlant,
        graph: Graph,
        replay_buffer: ReplayBuffer,
        params: ParameterContainer,
        save_local: bool = True,
        load_local: bool = False,
        verbose: bool = True,
    ) -> None:
        """Initialize the Learner."""
        self.params = params
        self.plant = plant
        self.graph = graph
        if params.learner_use_planner_exploration:
            # uses its own graph to sample actions for current states (see planner_exploration function)
            self.action_sampler = ActionSampler(plant, Graph(plant, params), params)
            self.exploration_function: Optional[Callable] = self.planner_exploration
        else:
            self.exploration_function = None
        self.replay_buffer = replay_buffer
        size_s = plant.state_dimension
        size_a = plant.action_dimension
        self.actor = Actor(size_s * 2, size_a)
        self.actor_expert = Actor(size_s * 2, size_a)
        self.critic = Critic(size_s * 2, size_a)
        self.state_norm = Normalizer(size_s)
        self.planner_experience_share = params.learner_max_planner_experience_share
        # the relative distance required to complete the task
        self.final_success_distance = params.learner_final_success_distance
        # the relative distance required during curriculum learning
        self.current_success_distance = params.learner_initial_success_distance
        self.best_relative_distance = torch.inf
        self.inject_demonstration = True
        self.injections = 0
        self.verbose = verbose

        self.save_local = save_local
        self.load_local = load_local

        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        task = self.params.model_filename[:-4]

        base_path = str(Path(__file__).resolve().parent.parents[2])
        self.base_local_path = base_path + f"/examples/learning/models/{task}/"
        self.local_path = self.base_local_path + f"{now}/"

        if save_local:
            print("Local models uploaded to {}.".format(self.local_path))

        if load_local:
            model_path = find_latest_model_path(self.base_local_path)
            self.load_models(model_path)

    def actor_actions(
        self, actor: Actor, node_ids: IntTensor, action_time_step: float
    ) -> FloatTensor:
        sub_goals = self.replay_buffer.learning_goals[node_ids]
        with torch.no_grad():
            actor_actions = actor.select_action(
                self.state_norm,
                self.replay_buffer.states[node_ids],
                sub_goals,
                self.exploration_function,
            )
        return actor_actions * self.params.action_range * action_time_step

    def relative_distances_to(
        self,
        data_container: Union[Graph, ReplayBuffer],
        ids: IntTensor,
        target_states: FloatTensor,
    ) -> FloatTensor:
        node_distances = scaled_distances_to(
            self.plant, data_container.states[ids], target_states
        )
        root_ids = data_container.root_ids[ids]
        root_distance = scaled_distances_to(
            self.plant, data_container.states[root_ids], target_states
        )
        return node_distances / root_distance

    def reward_function(
        self,
        data_container: Union[Graph, ReplayBuffer],
        node_ids: FloatTensor,
        goals: FloatTensor,
    ) -> Tuple[FloatTensor, FloatTensor]:
        relative_distances = self.relative_distances_to(data_container, node_ids, goals)
        if self.params.learner_use_sparse_reward:
            return -(
                relative_distances > self.current_success_distance
            ).float(), relative_distances
        else:  # clipped relative distance as reward
            return -min(
                relative_distances, torch.ones_like(relative_distances)
            ), relative_distances

    def planner_exploration(self, root_states: FloatTensor) -> FloatTensor:
        params = self.action_sampler.params
        graph = self.action_sampler.graph
        plant = self.action_sampler.plant

        root_ids = torch.tensor(0).unsqueeze(0)
        root_states[:, plant.actuated_pos]
        root_start_actions = root_states[:, plant.actuated_pos]
        root_end_actions = root_states[:, plant.actuated_pos]
        root_relative_actions = torch.zeros((1, plant.action_dimension))
        graph.next_main_node_id = 0  # always overwrite the root
        graph.add_nodes(
            root_ids,
            root_ids,
            root_states,
            root_start_actions,
            root_end_actions,
            root_relative_actions,
        )

        relative_actions, _, _ = self.action_sampler(root_ids)
        action_normalization_scaling = params.action_range * params.action_time_step
        return relative_actions / action_normalization_scaling

    def update_norm(self, states: FloatTensor, goals: FloatTensor) -> None:
        """Update the normalizers with the current episode of play experience.

        Samples the trajectory instead of taking every experience to create a goal distribution that
        is equal to what the networks encouter.

        """
        self.state_norm.update(states)
        self.state_norm.update(goals)

    def policy_rollout(self, temporary: bool = False) -> Tuple[FloatTensor, bool]:
        params = self.params
        plant = self.plant
        replay_buffer = self.replay_buffer
        sub_goal_state = sample_random_goal_states(plant, params, 1)

        state = sample_random_start_states(plant, params)[0]
        action = state[self.plant.actuated_pos]
        parent_id = (
            replay_buffer.next_temporary_id
            if temporary
            else replay_buffer.next_learner_id
        )
        node_ids = replay_buffer.add_nodes(
            tensor([parent_id], dtype=int),
            tensor([parent_id], dtype=int),
            state.unsqueeze(0),
            action.unsqueeze(0),
            action.unsqueeze(0),
            torch.zeros_like(action).unsqueeze(0),
            temporary=temporary,
            sub_goal_state=sub_goal_state,
        )
        path_ids = torch.zeros(params.learner_trajectory_length, dtype=int)
        for t in range(params.learner_trajectory_length - 1):
            path_ids[t] = node_ids[0]

            if self.actor_expert.is_trained:
                states = self.replay_buffer.states[node_ids]
                goals = sub_goal_state.repeat(len(node_ids), 1)

                obs = self.state_norm.wrap_obs(states, goals)
                obs_expert = self.state_norm.wrap_obs(states, goals)

                actions = self.actor.target(obs)
                actions_expert = self.actor_expert.target(obs_expert)

                q = self.critic(obs, actions)
                q_expert = self.critic(obs_expert, actions_expert)

                if (
                    q >= q_expert
                ):  # Select action from best actor (retains noise in exploration)
                    relative_actions = self.actor_actions(
                        self.actor, node_ids, params.action_time_step
                    )
                    used_expert = False
                else:
                    relative_actions = self.actor_actions(
                        self.actor_expert, node_ids, params.action_time_step
                    )
                    used_expert = True
            else:
                relative_actions = self.actor_actions(
                    self.actor, node_ids, params.action_time_step
                )
                used_expert = False

            match self.params.action_start_mode:
                case ActionMode.RELATIVE_TO_CURRENT_STATE:
                    start_actions = replay_buffer.states[node_ids][
                        :, self.plant.actuated_pos
                    ]
                case ActionMode.RELATIVE_TO_PREVIOUS_END_ACTION:
                    start_actions = replay_buffer.end_actions[node_ids]
                case _:
                    print("Select a valid ActionMode for the params.action_start_mode.")
                    raise (NotImplementedError)
            start_actions = clip_actions(start_actions, params)

            match params.action_end_mode:
                case ActionMode.RELATIVE_TO_CURRENT_STATE:
                    actuated_pos = self.replay_buffer.states[node_ids][
                        :, self.plant.actuated_pos
                    ]
                    sampled_end_actions = actuated_pos + relative_actions
                case ActionMode.RELATIVE_TO_PREVIOUS_END_ACTION:
                    sampled_end_actions = (
                        self.replay_buffer.end_actions[node_ids] + relative_actions
                    )
                case ActionMode.ABSOLUTE_ACTION:
                    print("Select a valid ActionMode for the params.action_end_mode.")
                    raise (NotImplementedError)
            sampled_end_actions = clip_actions(sampled_end_actions, params)

            states = replay_buffer.states[node_ids]
            start_end_actions = torch.stack((start_actions, sampled_end_actions), dim=1)
            new_states, _ = self.plant.dynamics(
                states, start_end_actions, params.action_time_step
            )

            node_ids = replay_buffer.add_nodes(
                replay_buffer.root_ids[node_ids],
                node_ids,
                new_states,
                start_actions,
                sampled_end_actions,
                relative_actions,
                temporary=temporary,
                sub_goal_state=sub_goal_state,
            )

        path_ids[-1] = node_ids[0]

        return path_ids, used_expert

    def graph_rollout(self, temporary: bool = False) -> FloatTensor:
        params = self.params
        plant = self.plant
        graph = self.graph
        replay_buffer = self.replay_buffer

        graph.change_sub_goal_states(sample_random_goal_states(plant, params))

        best_id = graph.get_best_id(reward_based=False)
        path_to_goal = graph.shortest_path_to(best_id)[
            -params.learner_trajectory_length :
        ]
        # Padding by staying at the start state multiple times. Assumes stationary start state.
        padding_ids = torch.full(
            (params.learner_trajectory_length - len(path_to_goal),), path_to_goal[0]
        )
        path_ids = torch.concatenate((padding_ids, path_to_goal))

        root_ids = torch.full(
            (params.learner_trajectory_length,), replay_buffer.next_learner_id
        )
        parent_ids = torch.arange(
            replay_buffer.next_learner_id - 1,
            replay_buffer.next_learner_id + params.learner_trajectory_length - 1,
        )
        parent_ids[0] += 1  # first node is its own parent
        states = graph.states[path_ids]
        start_actions = graph.start_actions[path_ids]
        end_actions = graph.end_actions[path_ids]
        relative_actions = graph.relative_actions[path_ids]
        node_ids = self.replay_buffer.add_nodes(
            root_ids,
            parent_ids,
            states,
            start_actions,
            end_actions,
            relative_actions,
            temporary=temporary,
            sub_goal_state=self.graph.sub_goal_states[0],
        )

        return node_ids

    def set_demonstration_injection(self, final_success_rate: float) -> None:
        params = self.params

        if (
            params.learner_stop_converged_injection
            and self.final_success_distance > 0.5
        ):
            self.inject_demonstration = False
        else:
            match self.params.learner_injection_strategy:
                case 0:  # just initially
                    self.inject_demonstration = (
                        self.injections < params.learner_max_initial_injections
                    )
                    self.injections += self.inject_demonstration
                case 1:  # x% all the time
                    self.inject_demonstration = (
                        torch.rand(1).item() < self.planner_experience_share
                    )
                case 2:  # decreasing amount as performance improves
                    self.planner_experience_share = (
                        params.learner_max_planner_experience_share
                        * (1 - final_success_rate)
                    )
                    self.inject_demonstration = (
                        torch.rand(1).item() < self.planner_experience_share
                    )

    def train(self, num_epochs: int = 50) -> None:
        """Train a policy to solve the environment with DDPG.

        Trajectories are resampled with HER to solve sparse reward environments.

        `DDPG paper <https://arxiv.org/pdf/1509.02971.pdf>`_

        `HER paper <https://arxiv.org/pdf/1707.01495.pdf>`_
        """
        params = self.params
        replay_buffer = self.replay_buffer
        actor = self.actor
        actor_expert = self.actor_expert
        critic = self.critic
        state_norm = self.state_norm

        if self.verbose:
            print(f"training for {num_epochs} epochs (seed: {params.seed})")
            print(
                "epoch | final rate | succ. rate | succ. dist. | rel. dist. |",
                "demo share | exp. train. | exp. polic. | time",
            )

        final_success_rate = 0.0

        for epoch in range(num_epochs):
            epoch_start = time.perf_counter()
            expert_training_share = 0.0
            expert_policy_share = 0.0
            demo_share = 0.0

            actor.train()
            actor_expert.train()
            for _ in range(params.learner_cycles):
                for _ in range(params.learner_rollouts):
                    self.set_demonstration_injection(final_success_rate)
                    if self.inject_demonstration:
                        path_ids = self.graph_rollout()
                        demo_share += 1
                    else:
                        path_ids, used_expert = self.policy_rollout()
                        expert_policy_share += used_expert
                self.update_norm(
                    replay_buffer.states[path_ids],
                    replay_buffer.learning_goals[path_ids],
                )
                for _ in range(params.learner_batches):
                    expert_training_share += train_actor_critic(
                        actor,
                        actor_expert,
                        critic,
                        state_norm,
                        replay_buffer,
                        self.reward_function,
                        her_probability=params.learner_her_probability,
                    )
                actor.update_target()
                critic.update_target()
            actor.eval()
            actor_expert.eval()

            demo_share /= params.learner_cycles * params.learner_rollouts
            expert_training_share /= params.learner_cycles * params.learner_batches
            expert_policy_share /= params.learner_cycles * params.learner_rollouts
            epoch_end = time.perf_counter()
            final_success_rate, current_success_rate, relative_distance = (
                self.eval_agent()
            )
            epoch_time = epoch_end - epoch_start

            if self.verbose:
                print(
                    f"{epoch:5} | {final_success_rate:10.3f} | {current_success_rate:10.3f} |"
                    f" {self.current_success_distance:11.3f} | {relative_distance:10.3f} |"
                    f" {demo_share:10.3f} |"
                    f" {expert_training_share:11.3f} | {expert_policy_share:11.3f} | {epoch_time:.3f}s"
                )

            if relative_distance < self.best_relative_distance:
                self.best_relative_distance = relative_distance
                if self.save_local:
                    self.save_models()

            if (
                current_success_rate >= params.learner_early_stop
                and self.current_success_distance <= self.final_success_distance
            ):
                return
            else:  # Update hyperparameters
                if current_success_rate >= 0.5:
                    self.current_success_distance = max(
                        self.final_success_distance, self.current_success_distance - 0.1
                    )
                if (
                    params.learner_stop_converged_random_exploration
                    and self.current_success_distance <= self.final_success_distance * 2
                ):
                    self.actor.eps = 0.0  # no more random exploration when policy is close to convergence

    def state_action_training_data(
        self,
        num_trajectories: int = 1000,
        discount_factor: float = 0.98,
    ) -> Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor]:
        params = self.params
        plant = self.plant
        graph = self.graph

        successful_trajectories = 0

        states_data = torch.zeros((0, plant.state_dimension))
        goals_data = torch.zeros((0, plant.state_dimension))
        actor_actions_data = torch.zeros((0, plant.action_dimension))
        q_values_data = torch.zeros((0, 1))
        trajectory_length_data = torch.zeros((0, 1))

        for _ in range(num_trajectories):
            search_index = torch.randint(
                0, params.num_parallel_searches, size=()
            ).item()
            graph.change_sub_goal_states(sample_random_goal_states(plant, params))
            best_id = graph.get_best_id(
                reward_based=False, search_indices=torch.tensor([search_index])
            ).item()
            root_id = graph.root_ids[best_id]
            path = graph.shortest_path_to(best_id, start_id=root_id)[
                -params.learner_trajectory_length :
            ]
            if len(path) == 1:
                continue

            states = graph.states[path[0:-1]]  # current states
            relative_actions = graph.relative_actions[
                path[1:]
            ]  # current actions are stored in the next state
            actor_actions = (
                relative_actions / params.action_range / params.action_time_step
            )
            goals = torch.tile(
                graph.sub_goal_states, (len(path[1:]), 1)
            )  # achieved goals
            q_values = self.reward_function(graph, path[1:], goals)[0][
                :, None
            ]  # initialize with achieved rewards
            trajectory_length = len(path)

            if q_values[-1] == 0:  # only use successful trajectories
                for i in range(len(q_values) - 2, -1, -1):
                    q_values[i] += (
                        discount_factor * q_values[i + 1]
                    )  # update with discounted future q-value

                self.update_norm(graph.states[path], goals)

                states_data = torch.cat((states_data, states.clone()))
                goals_data = torch.cat((goals_data, goals.clone()))
                actor_actions_data = torch.cat(
                    (actor_actions_data, actor_actions.clone())
                )
                q_values_data = torch.cat((q_values_data, q_values.clone()))
                trajectory_length_data = torch.tensor([trajectory_length], dtype=float)

                successful_trajectories += 1
        print(f"Found {successful_trajectories} successful trajectories")

        return (
            states_data,
            goals_data,
            actor_actions_data,
            q_values_data,
            trajectory_length_data,
        )

    def pretrain(
        self,
        num_epochs: int = 100,
        num_trajectories: int = 1000,
        train_critic: bool = True,
    ) -> None:
        actor = self.actor_expert
        actor.is_trained = True
        critic = self.critic
        state_norm = self.state_norm

        old_success_distance = self.current_success_distance
        self.current_success_distance = self.final_success_distance
        states, goals, actor_actions, q_values, trajectory_lengths = (
            self.state_action_training_data(num_trajectories=num_trajectories)
        )
        self.current_success_distance = old_success_distance

        if len(states) == 0:
            print("Not pretraining since no trajectories were found")
            return

        if self.verbose:
            print(f"pretraining for {num_epochs} epochs")
            print("epoch | actor loss | critic loss")

        self.actor_expert.train()
        for epoch in range(num_epochs):
            for _ in range(self.params.learner_batches):
                actor_loss = train_actor_imitation(
                    actor, state_norm, states, goals, actor_actions
                )
                if train_critic:
                    critic_loss = train_critic_imitation(
                        critic, state_norm, states, goals, actor_actions, q_values
                    )
                else:
                    critic_loss = 0.0
            if self.verbose:
                print(f"{epoch:5} | {actor_loss:10.3f} | {critic_loss:.3f}")
        self.actor_expert.eval()

        if self.save_local:
            self.save_models()

    def eval_agent(self) -> Tuple[float, float, float]:
        """Evaluate the current agent performance on the task.

        Runs `learner_evals` times and averages the success rate.
        """
        params = self.params
        learner_evals = params.learner_evals
        final_ids = torch.zeros(learner_evals, dtype=int)
        sub_goals = torch.zeros((learner_evals, self.plant.state_dimension))
        self.replay_buffer.reset_next_temporary_id()
        for i in range(learner_evals):
            final_ids[i] = self.policy_rollout(temporary=True)[0][-1]
            sub_goals[i] = self.replay_buffer.learning_goals[
                self.replay_buffer.next_temporary_id - 1
            ].clone()
        rewards, relative_distances = self.reward_function(
            self.replay_buffer, final_ids, sub_goals
        )
        final_success = torch.sum(
            1 - (relative_distances > self.final_success_distance).float()
        )
        current_success = torch.sum(1 + rewards)  # reward is -1 or 0
        relative_distance = torch.mean(relative_distances)

        return (
            final_success / learner_evals,
            current_success / learner_evals,
            relative_distance,
        )

    def save_models(self) -> None:
        """Save the actor and critic networks and the normalizers.

        Saves are located under `/models/<model_filename>/`.
        """
        path = self.local_path
        save_model(self.actor, path + "actor.pt")
        save_model(self.actor_expert, path + "actor_expert.pt")
        save_model(self.critic, path + "critic.pt")
        save_model(self.state_norm, path + "state_norm.pt")

    def load_models(self, path: str) -> None:
        """Load the actor and critic networks and the normalizers."""
        load_model(self.actor, path + "actor.pt")
        load_model(self.actor_expert, path + "actor_expert.pt")
        load_model(self.critic, path + "critic.pt")
        load_model(self.state_norm, path + "state_norm.pt")
