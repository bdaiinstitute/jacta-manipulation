# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import time
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from torch import FloatTensor

from jacta.planner.core.action_sampler import ActionSampler
from jacta.planner.core.graph import Graph
from jacta.planner.core.graph_worker import GraphWorker
from jacta.planner.core.logger import Logger
from jacta.planner.core.parameter_container import ParameterContainer
from jacta.planner.dynamics.simulator_plant import SimulatorPlant


class Planner:
    def __init__(
        self,
        plant: SimulatorPlant,
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
        self.params.reset_seed()
        self.plant.reset()
        self.graph.reset()
        self.logger.reset()
        self.action_sampler.reset()
        self.graph_worker.reset()

    def search(self) -> None:
        """Searches through the space for a trajectory to the final pose."""
        if self.verbose:
            print(
                f"searching with {self.params.steps_per_goal} steps",
                f"for each of the {self.params.num_sub_goals+1} goals (seed: {self.params.seed})",
            )
            print(
                "iterations | relative distance | scaled distance || success | finished | total ||"
            )

        t0 = time.time()

        # Initial check if goal is already reached
        if not self.graph_worker.callback_and_progress_check(
            -1, 100, verbose=self.verbose
        ).all():
            self.graph_worker.work(verbose=self.verbose)

        self.logger.total_time = time.time() - t0
        self.logger.create_distance_log()
        self.logger.create_reward_log()
        if self.verbose:
            print("dynamics computation time = ", round(self.logger.dynamics_time, 2))
            print("total search time = ", round(self.logger.total_time, 2))
            print(
                "pruned from",
                self.graph.next_main_node_id,
                "nodes to",
                self.graph.number_of_nodes(),
            )
            amortized_compute = round(
                self.logger.dynamics_time / self.graph.number_of_nodes(), 5
            )
            print("amortized dynamics time = ", amortized_compute)
            self.logger.simple_progress_statistics()

    def path_data(
        self, start_id: int, end_id: int
    ) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:
        """Returns the states, actions, and action time steps on the shortest path from
        start_id to end_id.
        """
        graph = self.graph

        path = graph.shortest_path_to(end_id, start_id=start_id)

        states = graph.states[path]
        start_actions = graph.start_actions[path]
        end_actions = graph.end_actions[path]

        return states, start_actions, end_actions

    def shortest_path_data(
        self, search_index: int = 0
    ) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:
        """Returns the states, actions, and action time steps on the shortest path from
        the root to the node closest to the goal.
        """
        graph = self.graph
        best_id = graph.get_best_id(
            reward_based=False, search_indices=torch.tensor([search_index])
        ).item()
        root_id = graph.root_ids[best_id]
        return self.path_data(root_id, best_id)

    def path_trajectory(
        self, path_data: Tuple[FloatTensor, FloatTensor, FloatTensor]
    ) -> FloatTensor:
        """Returns the trajectory for path_data."""
        states, start_actions, end_actions = path_data

        if len(states) == 1:
            trajectory = states
        else:
            trajectory = torch.zeros((0, self.plant.state_dimension))

            start_actions = start_actions[1:]
            end_actions = end_actions[1:]
            for i in range(len(start_actions)):
                actions_i = torch.vstack((start_actions[i], end_actions[i])).unsqueeze(
                    0
                )
                _, sub_trajectory = self.plant.dynamics(
                    states[i : i + 1], actions_i, self.params.action_time_step
                )
                if sub_trajectory.ndim == 3:  # n_states, num_substeps, nx
                    sub_trajectory = sub_trajectory.squeeze(0)
                trajectory = torch.cat((trajectory, sub_trajectory), dim=0)

        return trajectory

    def shortest_path_trajectory(self, search_index: int = 0) -> FloatTensor:
        """Returns the trajectory on the shortest path from the root to the node closest to the goal."""
        return self.path_trajectory(self.shortest_path_data(search_index=search_index))

    def plot_search_results(self) -> None:
        # plotting search results
        log_distances = self.logger.log_distances.cpu().numpy()
        plt.plot(
            log_distances[:, 0] / (self.graph.next_main_node_id - 1),
            label="closest node id",
        )
        plt.plot(log_distances[:, 1] / log_distances[0, 1], label="distance to goal")
        plt.xlim(0, self.graph.number_of_nodes())
        plt.ylim(0, 1)
        plt.legend()
        plt.show()
