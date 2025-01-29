# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
"""Allows the user to benchmark planner instances."""

from typing import Optional

import torch
from torch import IntTensor

from jactamanipulation.planner.planner.planner import Planner
from jactamanipulation.planner.visuals.mujoco_visualizer import MujocoRenderer


class Benchmark:
    """Benchmark class."""

    def __init__(
        self,
        planners: list[Planner],
        seeds: Optional[IntTensor] = None,
        names: Optional[list] = None,
        verbose: bool = False,
    ) -> None:
        """Constructs a benchmark object.

        Args:
            planners: list of planner objects to run
            seeds: random seeds to use on each planner
            names: for printing, unique name of planner
            verbose: verbose mode for planner

        """
        self.planners = planners
        number_of_planners = len(planners)
        if seeds is None:
            seeds = torch.zeros((number_of_planners, 1), dtype=int)
        self.seeds = seeds
        if names is None:
            names = [str(i) for i in range(number_of_planners)]
        self.names = names
        self.results_distance = torch.zeros(number_of_planners)
        self.results_timing = torch.zeros(number_of_planners)
        self.results_nodes_pre_prune = torch.zeros(number_of_planners, dtype=int)
        self.results_nodes_post_prune = torch.zeros(number_of_planners, dtype=int)
        self.results_trajectory_length = torch.zeros(number_of_planners, dtype=int)
        self.verbose = verbose

    def run_benchmark(self) -> None:
        """Iterates over planners and runs them with each seed sequentially."""
        for i, planner in enumerate(self.planners):
            print("Running planner", self.names[i])
            planner.verbose = self.verbose
            runs = len(self.seeds[i])
            for planner_seed in self.seeds[i]:
                print("  seed", planner_seed)
                planner.params.seed = int(planner_seed)
                planner.reset()
                graph = planner.graph

                try:
                    planner.search()
                except RuntimeError:
                    print("search errored for seed", planner_seed)
                states, _, _ = planner.shortest_path_data()

                search_index = 0  # only log the first search
                best_id = graph.get_best_id(reward_based=False)[search_index]
                best_distance = graph.scaled_goal_distances[best_id]
                root_id = graph.root_ids[best_id]
                root_distance = graph.scaled_goal_distances[root_id]
                self.results_distance[i] += (best_distance / root_distance) / runs
                self.results_timing[i] += planner.logger.total_time / runs
                self.results_nodes_pre_prune[i] += round((graph.next_main_node_id) / runs)
                self.results_nodes_post_prune[i] += round(graph.number_of_nodes() / runs)
                self.results_trajectory_length[i] += round(len(states) / runs)

    def print_results(self) -> None:
        """Prints distance, timing, pre-prune nodes, post-prune nodes, and trajectory length of benchmarks.

        Note:
            Run after calling ``run_benchmark``.
        """
        length_name = max([len(name) for name in ["planner", *self.names]])
        length_distance = len("distance")
        length_timing = len("timing")
        length_pre_prune = len("pre prune")
        length_post_prune = len("post prune")
        length_trajectory_length = len("traj length")
        print(
            "planner",
            " " * (length_name - len("planner")),
            " | distance | timing | pre prune | post prune | traj length",
            sep="",
        )
        print(
            "-" * length_name,
            "-|-",
            "-" * length_distance,
            "-|-",
            "-" * length_timing,
            "-|-",
            "-" * length_pre_prune,
            "-|-",
            "-" * length_post_prune,
            "-|-",
            "-" * length_trajectory_length,
            sep="",
        )
        for i, name in enumerate(self.names):
            distance = str(round(self.results_distance[i].item(), 3))
            timing = str(round(self.results_timing[i].item(), 2))
            pre_prune = str(round(self.results_nodes_pre_prune[i].item(), 1))
            post_prune = str(round(self.results_nodes_post_prune[i].item(), 1))
            trajectory_length = str(round(self.results_trajectory_length[i].item(), 1))

            print(
                name,
                " " * (length_name - len(name)),
                " | ",
                distance,
                " " * (length_distance - len(distance)),
                " | ",
                timing,
                " " * (length_timing - len(timing)),
                " | ",
                pre_prune,
                " " * (length_pre_prune - len(pre_prune)),
                " | ",
                post_prune,
                " " * (length_post_prune - len(post_prune)),
                " | ",
                trajectory_length,
                " " * (length_trajectory_length - len(trajectory_length)),
                sep="",
            )

    def visualize_results(self, planner_indices: Optional[IntTensor] = None) -> None:
        """Calls meshcat visualization of benchmark results.

        Note:
            Run after calling ``run_benchmark``.
        """
        if planner_indices is None:
            planner_indices = [i for i in range(len(self.planners))]

        for i in planner_indices:
            planner = self.planners[i]
            renderer = MujocoRenderer(plant=planner.plant)
            state_trajectory = planner.shortest_path_trajectory(search_index=0)
            planner.plot_search_results()
            renderer.show(state_trajectory, goal=planner.graph.goal_states[0])
