# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import logging
import time
from typing import Optional, Tuple

import torch
from torch import tensor

from jacta.planner.core.graph import Graph
from jacta.planner.core.parameter_container import ParameterContainer
from jacta.planner.core.types import ActionType, SelectionType


class Logger:
    def __init__(
        self,
        graph: Graph,
        params: ParameterContainer,
        search_index: int = 0,
        log_path: str = "/workspaces/bdai/projects/jacta/log/",
        log_file: Optional[str] = None,
    ):
        self.search_index = search_index
        self._initialize(graph, params, log_path, log_file)

    def reset(self) -> None:
        self._initialize(self.graph, self.params)

    def _initialize(
        self,
        graph: Graph,
        params: ParameterContainer,
        log_path: str = "/workspaces/bdai/projects/jacta/log/",
        log_file: Optional[str] = None,
    ) -> None:
        self.params = params
        self.graph = graph

        self.dynamics_time = 0.0
        self.total_time = 0.0
        self.temporary_time = 0.0

        self.iterations_list: list = []
        self.log_filename = None
        if params.log_to_file:
            if log_file is None:
                est_start_time = self._format_datetime(time.localtime())
                log_file = est_start_time + ".log"
            log_filename = log_path + log_file
            logging.basicConfig(
                filename=log_filename,
                filemode="w",
                format="%(message)s",
                level=logging.INFO,
                force=True,
            )
            self.log_filename = log_filename

            self.log_params()

    def get_log_name(self) -> Optional[str]:
        """Get where the log is stored"""
        return self.log_filename

    def _format_datetime(self, tm: time.struct_time) -> str:
        hour = (tm.tm_hour - 4) % 24
        return f"{tm.tm_mon}_{tm.tm_mday}_{tm.tm_year}_{hour}_{tm.tm_min}_{tm.tm_sec}"

    def log_params(self) -> None:
        log_string = (
            "planner parameters:\n"
            f"  model: {self.params.model_filename}\n"
            f"  reward_based: {self.params.reward_based}\n"
            f"  intermediate_pruning: {self.params.intermediate_pruning}"
        )
        logging.info(log_string)

    def log_search(self, iteration_number: int) -> None:
        log_string = f"iteration {iteration_number}:"
        logging.info(log_string)

        self.iterations_list.append([])

        self._reset_intermediate_time()

    def _log_values(self, type: str, keys: list, values: list) -> None:
        elapsed_time = self._get_intermediate_time()
        self.iterations_list[-1].append({})

        log_string = f"  {type}:\n"
        self.iterations_list[-1][-1]["type"] = type

        for i, key in enumerate(keys):
            log_string += f"    {key}: {values[i]}\n"
            self.iterations_list[-1][-1][key] = values[i]

        log_string += f"    time: {elapsed_time}"
        self.iterations_list[-1][-1]["time"] = elapsed_time

        if self.params.log_to_file:
            logging.info(log_string)

    def log_node_selection(
        self, node_ids: torch.IntTensor, strategy: SelectionType
    ) -> None:
        self._log_values(
            "node_selection", ["node_ids", "strategy"], [node_ids, strategy]
        )
        self._reset_intermediate_time()

    def log_action_sampler(
        self, node_ids: torch.IntTensor, strategy: ActionType
    ) -> None:
        self._log_values(
            "action_sampler", ["node_ids", "strategy"], [node_ids, strategy]
        )
        self._reset_intermediate_time()

    def log_node_extension(
        self, node_ids: torch.IntTensor, best_id: torch.IntTensor, dynamics_time: float
    ) -> None:
        self.dynamics_time += dynamics_time
        search_indices = self.graph.node_id_to_search_index_map[node_ids]
        node_ids = node_ids[search_indices == self.search_index]
        is_best_node = node_ids == best_id
        self._log_values(
            "node_extension",
            [
                "node_ids",
                "scaled_goal_distance",
                "value",
                "is_best_node",
                "dynamics_time",
            ],
            [
                node_ids,
                self.graph.scaled_goal_distances[node_ids],
                self.graph.rewards[node_ids],
                is_best_node,
                dynamics_time,
            ],
        )
        self._reset_intermediate_time()

    def log_node_pruning(
        self, node_id: int, strategy: str, num_removed_nodes: int
    ) -> None:
        self._log_values(
            "node_pruning",
            ["node_id", "strategy", "num_removed_nodes"],
            [node_id, strategy, num_removed_nodes],
        )
        self._reset_intermediate_time()

    def create_distance_log(self) -> None:
        graph = self.graph

        valid_node_ids = graph.get_active_main_ids(search_index=self.search_index)
        self.log_distances = torch.zeros(len(valid_node_ids), 2)
        for index, node_id in enumerate(valid_node_ids):
            new_distance = graph.scaled_goal_distances[node_id]
            if new_distance < self.log_distances[-1, 1]:
                self.log_distances[index] = tensor([node_id, new_distance])
            else:
                self.log_distances[index] = self.log_distances[index - 1]

    def create_reward_log(self) -> None:
        graph = self.graph

        valid_node_ids = graph.get_active_main_ids(search_index=self.search_index)
        self.log_rewards = torch.zeros(len(valid_node_ids), 2)
        for index, node_id in enumerate(valid_node_ids):
            new_reward = graph.rewards[node_id]
            if new_reward > self.log_rewards[-1, 1]:
                self.log_rewards[index] = tensor([node_id, new_reward])
            else:
                self.log_rewards[index] = self.log_rewards[index - 1]

    def _reset_intermediate_time(self) -> None:
        self.temporary_time = time.perf_counter()

    def _get_intermediate_time(self) -> float:
        return time.perf_counter() - self.temporary_time

    def simple_progress_statistics(self) -> None:
        # [a,b]: a=best number, b=total number
        selection_strategies = {type: tensor([0, 0]) for type in SelectionType}
        action_strategies = {type: tensor([0, 0]) for type in ActionType}

        total_selections = 0
        total_actions = 0

        for iteration in self.iterations_list:
            for i, operation in enumerate(iteration):
                operation_type = operation["type"]
                if operation_type == "node_selection":
                    total_selections += 1
                    selection_strategy = operation["strategy"]
                    selection_strategies[selection_strategy][1] += 1
                elif operation_type == "action_sampler":
                    total_actions += 1
                    action_strategy = operation["strategy"]
                    action_strategies[action_strategy][1] += 1
                    next_operation = iteration[i + 1]
                    assert next_operation["type"] == "node_extension"
                    if any(next_operation["is_best_node"]):
                        selection_strategies[selection_strategy][0] += 1
                        action_strategies[action_strategy][0] += 1

        print("")
        print(f"Selection strategies ({total_selections})")
        print("Name     | Total share | Own best share")
        for name in SelectionType:
            total_share = torch.round(
                100 * selection_strategies[name][1] / total_selections
            )
            progress_share_rel = (
                selection_strategies[name][0] / selection_strategies[name][1]
            )
            progress_share_abs = selection_strategies[name][0]
            print(
                f"{name:8} | {total_share:10}% |"
                f" {progress_share_rel:.1e} ({progress_share_abs}/{selection_strategies[name][1]})"
            )

        print("")
        print(f"Action strategies ({total_actions})")
        print("Name          | Total share | Own best share")
        for name in ActionType:
            total_share = torch.round(100 * action_strategies[name][1] / total_actions)
            progress_share_rel = action_strategies[name][0] / action_strategies[name][1]
            progress_share_abs = action_strategies[name][0]
            print(
                f"{name:13} | {total_share:10}% |"
                f" {progress_share_rel:.1e} ({progress_share_abs}/{action_strategies[name][1]})"
            )

    def simple_path_statistics(self) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        idx = self.graph.get_best_id(
            reward_based=False, search_indices=torch.tensor([self.search_index])
        ).item()
        path_to_goal = self.graph.shortest_path_to(idx)
        num_edges = len(path_to_goal) - 1
        edge_starts = path_to_goal[:-1]
        edge_ends = path_to_goal[1:]

        selection_strategies = ["" for _ in range(num_edges)]
        action_strategies = ["" for _ in range(num_edges)]

        for iteration in self.iterations_list:
            for i, operation in enumerate(iteration):
                operation_type = operation["type"]
                if operation_type == "node_selection":
                    selection_strategy = operation["strategy"]
                elif operation_type == "action_sampler":
                    node_ids = operation["node_ids"]
                    idx = torch.nonzero(
                        tensor([node_id in edge_starts for node_id in node_ids])
                    )
                    if idx.numel() == 0:
                        break
                    else:
                        idx = idx[0, 0].item()
                        action_strategy = operation["strategy"]
                        start_id = node_ids[idx]
                        edge_index = (edge_starts == start_id).nonzero().squeeze()
                        next_operation = iteration[i + 1]
                        assert next_operation["type"] == "node_extension"
                        if next_operation["node_ids"][idx] == edge_ends[edge_index]:
                            action_strategies[edge_index] = str(action_strategy)
                            selection_strategies[edge_index] = str(selection_strategy)

        return selection_strategies, action_strategies
