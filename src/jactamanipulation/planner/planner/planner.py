# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import time
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from torch import FloatTensor

from jactamanipulation.planner.dynamics.locomotion_plant import LocomotionPlant
from jactamanipulation.planner.dynamics.mujoco_dynamics import MujocoPlant
from jactamanipulation.planner.planner.action_sampler import ActionSampler
from jactamanipulation.planner.planner.graph import Graph
from jactamanipulation.planner.planner.graph_worker import GraphWorker
from jactamanipulation.planner.planner.logger import Logger
from jactamanipulation.planner.planner.parameter_container import ParameterContainer
from jactamanipulation.planner.planner.types import ActionType


class Planner:
    """Planner"""

    def __init__(
        self,
        plant: MujocoPlant,
        graph: Graph,
        action_sampler: ActionSampler,
        graph_worker: GraphWorker,
        logger: Logger,
        params: ParameterContainer,
        verbose: bool = False,
    ) -> None:
        self.params = params
        self.plant = plant
        self.graph = graph
        self.logger = logger
        self.action_sampler = action_sampler
        self.graph_worker = graph_worker

        self.verbose = verbose

    def reset(self) -> None:
        """Reset"""
        self.params.reset_seed()
        self.plant.reset()
        self.graph.reset()
        self.logger.reset()
        self.action_sampler.reset()
        self.graph_worker.reset()

    def search(self) -> None:
        """Searches through the space for a trajectory to the goal state."""
        if self.verbose:
            print(
                f"searching with {self.params.steps_per_goal} steps",
                f"for each of the {self.params.num_sub_goals+1} goals (seed: {self.params.seed})",
            )
            print("iterations | relative distance | scaled distance || success | finished | total ||")

        t0 = time.time()

        # Initial check if goal is already reached
        if not self.graph_worker.callback_and_progress_check(iteration=-1, num_steps=100, verbose=self.verbose).all():
            self.graph_worker.work(verbose=self.verbose)

        self.logger.total_time = time.time() - t0
        self.logger.create_distance_log()
        self.logger.create_reward_log()
        if self.verbose:
            print("dynamics computation time = ", round(self.logger.dynamics_time, 2))
            print("total search time = ", round(self.logger.total_time, 2))
            print("pruned from", self.graph.next_main_node_id, "nodes to", self.graph.number_of_nodes())
            dynamics_time_per_node = round(self.logger.dynamics_time / self.graph.next_main_node_id + 1, 5)
            print("dynamics time per main node = ", dynamics_time_per_node)
            self.logger.simple_progress_statistics()

    def path_data(self, start_id: int, end_id: int) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:
        """Returns the states, start and end actions, on the shortest path from start_id to end_id

        Returns the states, start actions, and end actions on the shortest path from
        start_id to end_id.
        """
        graph = self.graph

        path = graph.shortest_path_to(end_id, start_id=start_id)

        states = graph.states[path]
        end_actions = graph.end_actions[path]
        relative_actions = graph.relative_actions[path]

        return states, end_actions, relative_actions

    def shortest_path_data(self, search_index: int = 0) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:
        """Returns the states, actions, and action time steps on the shortest path

        Returns the states, actions, and action time steps on the shortest path from
        the root to the node closest to the goal.
        """
        graph = self.graph
        best_id = graph.get_best_id(reward_based=False, search_indices=torch.tensor([search_index])).item()
        root_id = graph.root_ids[best_id]
        return self.path_data(root_id, best_id)

    def path_trajectory(self, path_data: Tuple[FloatTensor, FloatTensor, FloatTensor]) -> FloatTensor:
        """Returns the trajectory for path_data."""
        states, end_actions, relative_actions = path_data

        if len(states) == 1:
            trajectory = states
        else:
            trajectory = states[0:1]

            info = {}
            if isinstance(self.plant, LocomotionPlant):
                info["sensor"] = self.plant.get_sensor(states[0:1])

            previous_end_actions = end_actions[0:-1]
            relative_actions = relative_actions[1:]

            for i in range(len(relative_actions)):
                _, _, start_end_sub_actions = self.graph_worker.action_processor(
                    relative_actions=relative_actions[i : i + 1],
                    action_type=ActionType.NON_EXPERT,
                    current_states=states[i : i + 1],
                    previous_end_actions=previous_end_actions[i : i + 1],
                )
                *_, sensor, sub_trajectory = self.plant.dynamics(states[i : i + 1], start_end_sub_actions, info)
                info["sensor"] = sensor
                if sub_trajectory.ndim == 3:  # n_states, num_substeps, nx
                    sub_trajectory = sub_trajectory.squeeze(0)
                trajectory = torch.cat((trajectory, sub_trajectory), dim=0)
        return trajectory

    def shortest_path_trajectory(self, search_index: int = 0) -> FloatTensor:
        """Returns the trajectory on the shortest path from the root to the node closest to the goal."""
        return self.path_trajectory(self.shortest_path_data(search_index=search_index))

    def plot_search_results(self) -> None:
        """Plot search results"""
        search_progress = self.logger.search_progress.cpu().numpy()
        plt.plot(search_progress[:, 0] / (self.graph.next_main_node_id - 1), label="closest node id")
        plt.plot(search_progress[:, 1] / search_progress[0, 1], label="distance to goal")
        plt.xlim(0, self.graph.number_of_nodes())
        plt.ylim(0, 1)
        plt.legend()
        plt.show()
