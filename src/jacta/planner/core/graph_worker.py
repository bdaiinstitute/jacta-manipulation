# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import math
import time
from typing import Callable, Optional, Tuple

import torch
from torch import BoolTensor, FloatTensor, IntTensor

from jacta.planner.core.action_sampler import ActionSampler
from jacta.planner.core.clipping_method import clip_actions
from jacta.planner.core.graph import (
    Graph,
    sample_random_sub_goal_states,
    sample_related_sub_goal_states,
)
from jacta.planner.core.linear_algebra import (
    project_vectors_on_eigenspace,
    truncpareto_cdf,
)
from jacta.planner.core.logger import Logger
from jacta.planner.core.parameter_container import ParameterContainer
from jacta.planner.core.types import (
    ACTION_TYPE_DIRECTIONAL_MAP,
    ActionMode,
    ActionType,
    SelectionType,
)
from jacta.planner.dynamics.simulator_plant import SimulatorPlant, scaled_distances_to


def pareto_distribution(length: int, exponent: float) -> FloatTensor:
    upper_bound = length + 1
    x = torch.arange(1, upper_bound)
    xU = x + 1
    distribution = truncpareto_cdf(xU, exponent, upper_bound) - truncpareto_cdf(
        x, exponent, upper_bound
    )
    return distribution / distribution.sum()


class GraphWorker:
    def __init__(
        self,
        plant: SimulatorPlant,
        graph: Graph,
        action_sampler: ActionSampler,
        logger: Logger,
        params: ParameterContainer,
        callback: Optional[Callable] = None,
        callback_period: Optional[
            int
        ] = None,  # period in number of steps which dictates when the callback is called
    ):
        self._initialize(
            plant, graph, action_sampler, logger, params, callback, callback_period
        )

    def reset(self) -> None:
        self._initialize(
            self.plant,
            self.graph,
            self.action_sampler,
            self.logger,
            self.params,
            self.callback,
            self.callback_period,
        )

    def _initialize(
        self,
        plant: SimulatorPlant,
        graph: Graph,
        action_sampler: ActionSampler,
        logger: Logger,
        params: ParameterContainer,
        callback: Optional[Callable] = None,
        callback_period: Optional[int] = None,
    ) -> None:
        self.params = params
        self.plant = plant
        self.graph = graph
        self.logger = logger
        self.action_sampler = action_sampler
        self.callback = callback
        self.callback_period = callback_period
        self.search_succeeded = torch.zeros(
            params.num_parallel_searches, dtype=torch.bool
        )
        self.search_finished = torch.zeros(
            params.num_parallel_searches, dtype=torch.bool
        )

        # Pareto distribution parameters
        self.extension_horizon = torch.ones(params.num_parallel_searches)
        self.pareto_exponent = (
            torch.ones(params.num_parallel_searches) * params.pareto_exponent_max
        )

    def node_selection(self, search_indices: IntTensor) -> IntTensor:
        """Selects a collection of nodes. Nodes a ranked either by reward or scaled distance to goal.
        Then nodes are selected according to the Pareto distribution.

        Args:
            search_indices: the indices of the searches to select nodes for
        """
        sampled_node_ids = torch.zeros_like(search_indices)
        unique_search_indices = torch.unique(search_indices)
        for search_index in unique_search_indices:
            sorted_ids = self.graph.sorted_progress_ids(
                self.params.reward_based, search_index
            )
            distribution = pareto_distribution(
                len(sorted_ids), self.pareto_exponent[search_index]
            )
            number_of_nodes = torch.sum(search_indices == search_index)
            replacement = (
                len(sorted_ids) < number_of_nodes
            ).item() or self.params.force_replace
            indices = distribution.multinomial(
                num_samples=number_of_nodes, replacement=replacement
            )
            sampled_node_ids[search_indices == search_index] = sorted_ids[indices]
        return sampled_node_ids

    def get_start_actions(self, node_ids: IntTensor) -> FloatTensor:
        match self.params.action_start_mode:
            case ActionMode.RELATIVE_TO_CURRENT_STATE:
                actions = self.graph.states[node_ids][:, self.plant.actuated_pos]
            case ActionMode.RELATIVE_TO_PREVIOUS_END_ACTION:
                actions = self.graph.end_actions[node_ids]
            case _:
                print("Select a valid ActionMode for the params.action_start_mode.")
                raise (NotImplementedError)
        actions = clip_actions(actions, self.params)

        return actions

    def get_end_actions(
        self,
        node_ids: IntTensor,
        relative_actions: FloatTensor,
        action_type: Optional[ActionType],
    ) -> FloatTensor:
        match self.params.action_end_mode:
            case ActionMode.RELATIVE_TO_CURRENT_STATE:
                actions = (
                    self.graph.states[node_ids][:, self.plant.actuated_pos]
                    + relative_actions
                )
            case ActionMode.RELATIVE_TO_PREVIOUS_END_ACTION:
                actions = self.graph.end_actions[node_ids] + relative_actions
            case ActionMode.ABSOLUTE_ACTION:
                actions = relative_actions
            case _:
                print("Select a valid ActionMode for the params.action_end_mode.")
                raise (NotImplementedError)
        actions = clip_actions(actions, self.params)

        is_directional = ACTION_TYPE_DIRECTIONAL_MAP.get(action_type, True)
        if not is_directional and self.params.using_eigenspaces:
            actions = project_vectors_on_eigenspace(
                actions, self.params.orthonormal_basis
            )

        return actions

    def node_extension(
        self,
        node_ids: IntTensor,
        relative_actions: FloatTensor,
        num_action_steps: int,
        action_type: Optional[ActionType] = None,
    ) -> Tuple[IntTensor, float, bool]:
        """Chooses a node to extend to based on the current node and action sampler.

        Args:
            node_ids: the id sof the nodes to extend from with the actions
            actions: control vectors of size (nu,)
            num_action_steps: the number of steps. Must be the same for all extensions to perform parallel rollout
        """
        graph = self.graph
        params = self.params

        # start dynamics rollout
        t0 = time.perf_counter()

        states = self.graph.states[node_ids]

        last_valid_main_ids = node_ids.clone()
        for step in range(num_action_steps):
            start_actions = self.get_start_actions(node_ids)
            end_actions = self.get_end_actions(node_ids, relative_actions, action_type)
            start_end_sub_actions = torch.stack((start_actions, end_actions), dim=1)

            states, _ = self.plant.dynamics(
                states, start_end_sub_actions, params.action_time_step
            )
            is_main_node = step == num_action_steps - 1

            node_ids, graph_full = graph.add_nodes(
                graph.root_ids[node_ids],
                node_ids,
                states,
                start_actions,
                end_actions,
                relative_actions,
                is_main_node,
            )
            if graph_full:
                break

        dynamics_time = time.perf_counter() - t0
        # end dynamics rollout

        if graph_full:
            # avoids breaking logging not exposing sub-node ids
            node_ids = last_valid_main_ids

        return node_ids, dynamics_time, graph_full

    def node_pruning(self, paths_ids: IntTensor) -> IntTensor:
        """Finds the best node in path_ids and removes all nodes after the best node"""
        best_indices = torch.argmax(self.graph.rewards[paths_ids], dim=-1)
        for i, best_index in enumerate(best_indices):
            self.graph.deactivate_nodes(paths_ids[i, best_index + 1 :])

        return best_indices

    def node_replacement(
        self, node_ids: IntTensor, paths_ids: IntTensor, best_indices: IntTensor
    ) -> Tuple[int, bool]:
        """Tries to replace the path from predecessor_node to node with a direct_node from predecessor_node"""
        graph = self.graph
        params = self.params

        replace_indices = torch.nonzero(best_indices > 1).squeeze(
            1
        )  # replace only if more than 2 nodes in a path
        if replace_indices.numel() > 0:
            num_action_steps = torch.clamp(
                torch.min(best_indices[replace_indices]), 0, params.action_steps_max
            ).item()
            replace_node_ids = node_ids[replace_indices]
            relative_actions = (
                graph.end_actions[replace_node_ids]
                - graph.start_actions[replace_node_ids]
            )
            direct_node_ids, _, graph_full = self.node_extension(
                paths_ids[replace_indices, 0], relative_actions, num_action_steps
            )
            direct_distances = scaled_distances_to(
                self.plant,
                graph.states[replace_node_ids],
                graph.states[direct_node_ids],
            )

            worse_indices = graph.is_worse_than(
                direct_node_ids, replace_node_ids
            )  # if worse, remove new direct node
            graph.deactivate_nodes(direct_node_ids[worse_indices])

            close_indices = (
                direct_distances < 1e-2
            ) * ~worse_indices  # if close and not worse, remove old node and path
            graph.deactivate_nodes(paths_ids[replace_indices[close_indices], 1:])

            # otherwise keep both nodes

            return torch.sum(close_indices), graph_full
        else:
            return 0, False

    def percentage_range(self, start: int, stop: int) -> range:
        if stop < 10:
            return range(start, stop + 1, 1)
        else:
            return range(start, stop + 1, (stop + 1) // 10)

    def get_progress_info(
        self,
        iteration: int,
        num_steps: int,
        print_percentage: bool = False,
        verbose: bool = False,
    ) -> FloatTensor:
        graph = self.graph

        best_ids = graph.get_best_id(reward_based=False)
        best_distances = graph.scaled_goal_distances[best_ids]
        root_ids = graph.root_ids[best_ids]
        root_distances = graph.scaled_goal_distances[root_ids]
        relative_distances = best_distances / root_distances

        if verbose and (
            print_percentage or iteration in self.percentage_range(0, num_steps)
        ):
            percentage_done = round(100 * iteration / num_steps)
            mean_relative_distance = relative_distances.mean().item()
            mean_best_distance = best_distances.mean().item()
            num_search_succeeded = torch.sum(self.search_succeeded).item()
            num_searches_completed = torch.sum(self.search_finished).item()
            searches_total = len(self.search_finished)
            print(
                f"{percentage_done:9}% | "
                f"{int(mean_relative_distance * 100):16}% | "
                f"{mean_best_distance:15.3f} || "
                f"{num_search_succeeded:7} | {num_searches_completed:8} | {searches_total:5} ||"
            )

        return relative_distances

    def callback_and_progress_check(
        self,
        iteration: int,
        num_steps: int,
        change_goal: bool = False,
        verbose: bool = False,
    ) -> BoolTensor:
        """Calls the search callback. Returns True if goal reached"""
        # user defined callback: e.g. a graph visualization function
        if self.callback_period is not None and self.callback is not None:
            if (
                iteration >= self.callback_period
                and iteration % self.callback_period == 0
            ):
                self.callback(self.graph, self.logger)

        # Check for actual goal reached and print progress info
        if change_goal:
            self.graph.reset_sub_goal_states()
        relative_distances = self.get_progress_info(
            iteration + 1, num_steps, verbose=verbose
        )
        return relative_distances < self.params.termination_distance


class SingleGoalWorker(GraphWorker):
    def work(self, verbose: bool = False) -> bool:
        """Tries to find a path to a single goal."""
        params = self.params
        logger = self.logger
        len_node_ids = params.parallel_extensions

        for search_step in range(params.steps_per_goal):
            logger.log_search(search_step)

            made_progress = False
            node_ids = self.node_selection(
                search_indices=torch.zeros(len_node_ids, dtype=torch.int64)
            )
            selection_type = SelectionType.PARETO

            extensions = min(
                params.extension_horizon_max, math.ceil(self.extension_horizon)
            )
            paths_ids = torch.zeros((len(node_ids), extensions + 1), dtype=torch.int64)
            paths_ids[:, 0] = node_ids

            for extension_step in range(extensions):
                logger.log_node_selection(node_ids, selection_type)
                relative_actions, num_action_steps, action_type = self.action_sampler(
                    node_ids
                )
                logger.log_action_sampler(node_ids, action_type)
                new_node_ids, dynamics_time, graph_full = self.node_extension(
                    node_ids, relative_actions, num_action_steps, action_type
                )
                logger.log_node_extension(
                    new_node_ids,
                    self.graph.get_best_id(reward_based=False),
                    dynamics_time,
                )
                paths_ids[:, extension_step + 1] = new_node_ids

                has_best_rewards = node_ids == self.graph.get_best_id(
                    params.reward_based
                )
                if not made_progress and any(has_best_rewards):
                    self.pareto_exponent[0] = (
                        params.pareto_exponent_max
                    )  # reset pareto variance (more greedy)
                    self.extension_horizon[0] = (
                        self.extension_horizon[0] * 0.95 + (extension_step + 1) * 0.05
                    )
                    made_progress = True

                node_ids = new_node_ids
                selection_type = SelectionType.LAST

                if graph_full:
                    break

            if not made_progress:
                # Takes ~175 unsuccessful steps to get from pareto_exponent_max to pareto_exponent_min
                self.pareto_exponent[0] = max(
                    self.pareto_exponent[0] * 0.99, params.pareto_exponent_min
                )  # Increase pareto variance (less greedy)
                self.extension_horizon[0] = (
                    self.extension_horizon[0] * 0.95 + (extensions + 1) * 0.05
                )  # Assume one more extension would have worked

            if params.intermediate_pruning and not graph_full:
                best_indices = self.node_pruning(paths_ids)
                logger.log_node_pruning(
                    paths_ids[:, -1],
                    "prune",
                    paths_ids[:, 1:].numel() - torch.sum(best_indices),
                )
                node_ids = paths_ids[torch.arange(paths_ids.shape[0]), best_indices]
            else:
                best_indices = torch.ones(len_node_ids, dtype=torch.int64) * extensions
            if params.intermediate_replacement and not graph_full:
                num_replaced_nodes, graph_full = self.node_replacement(
                    node_ids, paths_ids, best_indices
                )
                logger.log_node_pruning(node_ids, "replace", num_replaced_nodes)

            # do callbacks and termination checking here if we don't have sub goals
            if params.num_sub_goals == 0:
                if (
                    self.callback_and_progress_check(
                        search_step, params.steps_per_goal, verbose=verbose
                    )
                    or graph_full
                ):
                    self.get_progress_info(
                        search_step,
                        params.steps_per_goal,
                        print_percentage=True,
                        verbose=verbose,
                    )
                    break

        return graph_full


class ParallelGoalsWorker(GraphWorker):
    def __init__(self, *args: Tuple, **kwargs: dict):
        super().__init__(*args, **kwargs)  # type: ignore
        params = self.params

        if params.intermediate_pruning or params.intermediate_replacement:
            raise ValueError(
                "ParallelGoalsWorker does not support intermediate pruning or replacement"
            )

        self.num_workers = params.num_parallel_searches * params.parallel_extensions
        # parallel search / worker variables
        self.search_steps = torch.zeros(params.num_parallel_searches, dtype=torch.int64)
        self.search_extensions_step = torch.zeros(
            params.num_parallel_searches, dtype=torch.int64
        )
        self.search_extensions_length = torch.ones(
            params.num_parallel_searches, dtype=torch.int64
        )
        self.worker_paths_ids = -torch.ones(
            (self.num_workers, self.params.extension_horizon_max + 1), dtype=torch.int64
        )
        self.worker_search_indices = torch.floor(
            torch.linspace(0, params.num_parallel_searches - 0.01, self.num_workers)
        ).long()

    def try_to_reallocate_workers(self, worker_reset_mask: BoolTensor) -> None:
        # replaces search indices of the completed searches
        worker_reset_search_indices = self.worker_search_indices[worker_reset_mask]
        worker_search_completed_mask = self.search_finished[worker_reset_search_indices]
        if torch.any(worker_search_completed_mask):
            # get search indices
            uncompleted_search_indices = self.graph.search_indices[
                ~self.search_finished
            ]
            number_of_new_search_indices = torch.sum(
                worker_search_completed_mask
            ).item()
            new_search_index_ids = torch.floor(
                torch.linspace(
                    0,
                    len(uncompleted_search_indices) - 0.01,
                    number_of_new_search_indices,
                )
            ).long()
            new_search_indices = uncompleted_search_indices[new_search_index_ids]
            self.worker_search_indices[worker_search_completed_mask] = (
                new_search_indices
            )

    def update_extension_lengths(self, search_reset_mask: BoolTensor) -> None:
        new_extension_lengths = torch.ceil(self.extension_horizon[search_reset_mask])
        new_extension_lengths = torch.clamp(
            new_extension_lengths, 1, self.params.extension_horizon_max
        )
        self.search_extensions_length[search_reset_mask] = new_extension_lengths.long()
        self.search_extensions_step[search_reset_mask] = 0

    def reset_finished_workers(self) -> None:
        search_reset_mask = self.search_extensions_step >= (
            self.search_extensions_length - 1
        )
        if torch.any(search_reset_mask):
            reset_search_indices = self.graph.search_indices[search_reset_mask]
            worker_reset_mask = torch.isin(
                self.worker_search_indices, reset_search_indices
            )

            self.try_to_reallocate_workers(worker_reset_mask)
            self.update_extension_lengths(search_reset_mask)

            # select new node ids
            search_indices_to_expand = self.worker_search_indices[worker_reset_mask]
            new_node_ids = self.node_selection(search_indices=search_indices_to_expand)

            # reset path ids
            self.worker_paths_ids[worker_reset_mask, 0] = new_node_ids
            self.worker_paths_ids[
                worker_reset_mask, 1:
            ] = -1  # not necessary, but for clarity

    def update_pareto_parameters(
        self,
        node_ids: IntTensor,
        new_node_ids: IntTensor,
    ) -> None:
        params = self.params
        logger = self.logger
        best_ids = self.graph.get_best_id(params.reward_based)
        has_best_rewards = torch.isin(best_ids, new_node_ids)

        is_first_extension = self.search_extensions_step == 0

        pareto_update_mask = torch.logical_and(is_first_extension, has_best_rewards)
        if torch.any(pareto_update_mask):  # if clause is not necessary, but for clarity
            self.pareto_exponent[pareto_update_mask] = (
                params.pareto_exponent_max
            )  # reset pareto variance (more greedy)
            self.extension_horizon[pareto_update_mask] = (
                self.extension_horizon[pareto_update_mask] * 0.95
                + self.search_extensions_step[pareto_update_mask] * 0.05
            )  # Assume one more extension would have worked

        # log pareto selection
        if is_first_extension[logger.search_index]:
            selection_type = SelectionType.PARETO
        else:
            selection_type = SelectionType.LAST
        search_indices = self.graph.node_id_to_search_index_map[node_ids]
        search_0_node_ids = node_ids[search_indices == logger.search_index]
        logger.log_node_selection(search_0_node_ids, selection_type)

        is_last_extension = self.search_extensions_step == (
            self.search_extensions_length - 1
        )
        if torch.any(is_last_extension):
            made_progress = torch.logical_and(
                is_last_extension,
                self.pareto_exponent == params.pareto_exponent_max,
            )
            if torch.any(~made_progress):
                pareto_update_mask = ~made_progress
                self.pareto_exponent[pareto_update_mask] = torch.max(
                    self.pareto_exponent[pareto_update_mask] * 0.99,
                    params.pareto_exponent_min,
                )  # Increase pareto variance (less greedy)
                self.extension_horizon[pareto_update_mask] = (
                    self.extension_horizon[pareto_update_mask] * 0.95
                    + (self.search_extensions_length[pareto_update_mask] + 1) * 0.05
                )  # Assume one more extension would have worked

    def work(self, verbose: bool = False) -> bool:
        """Tries to find a path to a single goal."""
        params = self.params
        logger = self.logger

        least_advanced_search_step = 0
        while True:
            logger.log_search(least_advanced_search_step)

            self.reset_finished_workers()

            # get node ids
            worker_extension_step = self.search_extensions_step[
                self.worker_search_indices
            ]
            node_ids = torch.gather(
                self.worker_paths_ids, 1, worker_extension_step.unsqueeze(1)
            ).flatten()

            # get actions
            relative_actions, num_action_steps, action_type = self.action_sampler(
                node_ids
            )
            logger.log_action_sampler(node_ids, action_type)

            # extend nodes
            new_node_ids, dynamics_time, graph_full = self.node_extension(
                node_ids, relative_actions, num_action_steps, action_type
            )

            # update worker paths (used for pruning and replacement)
            self.worker_paths_ids[:, worker_extension_step + 1] = new_node_ids
            logger.log_node_extension(
                new_node_ids,
                self.graph.get_best_id(
                    reward_based=False,
                    search_indices=torch.tensor([logger.search_index]),
                ),
                dynamics_time,
            )

            self.update_pareto_parameters(node_ids, new_node_ids)

            # termination check
            self.search_succeeded = self.callback_and_progress_check(
                least_advanced_search_step, params.steps_per_goal, verbose=verbose
            )
            searches_out_of_time = self.search_steps >= params.steps_per_goal
            self.search_finished = torch.logical_or(
                self.search_succeeded, searches_out_of_time
            )
            if self.search_finished.all() or graph_full:
                self.get_progress_info(
                    least_advanced_search_step,
                    params.steps_per_goal,
                    print_percentage=True,
                    verbose=True,
                )
                break

            # update search steps
            finished_extensions = self.search_extensions_step == (
                self.search_extensions_length - 1
            )
            self.search_extensions_step += 1
            self.search_steps += finished_extensions
            least_advanced_search_step = torch.min(self.search_steps).item()

        return graph_full


class CommonGoalWorkerInterface:
    def __init__(self, *args: Tuple, **kwargs: dict):
        params: ParameterContainer = kwargs.get("params", args[4])
        worker_class: type = (
            ParallelGoalsWorker
            if params.num_parallel_searches > 1
            else SingleGoalWorker
        )
        self.parent = worker_class(*args, **kwargs)

        # expose common attributes
        self.graph = self.parent.graph
        self.params = self.parent.params
        self.logger = self.parent.logger

        # expose common methods
        self.reset = self.parent.reset
        self.callback_and_progress_check = self.parent.callback_and_progress_check
        self.get_progress_info = self.parent.get_progress_info


class RelatedGoalWorker(CommonGoalWorkerInterface):
    def work(self, verbose: bool = False) -> bool:
        """Tries to find paths to goals sampled around the actual goal."""
        num_sub_goals = self.params.num_sub_goals
        for search_step in range(num_sub_goals + 1):
            is_final_goal_turn = torch.rand(1).item() <= self.params.goal_bias
            if is_final_goal_turn or search_step == num_sub_goals:  # final goal
                self.graph.reset_sub_goal_states()
            else:
                new_sub_goal_states = sample_related_sub_goal_states(
                    self.params,
                    self.graph.start_states,
                    self.graph.goal_states,
                    self.params.num_parallel_searches,
                )
                self.graph.change_sub_goal_states(new_sub_goal_states)

            graph_full = self.parent.work(verbose=verbose)

            # do callbacks and termination checking here if we have sub goals
            if num_sub_goals > 0:
                if (
                    self.callback_and_progress_check(
                        search_step, num_sub_goals, change_goal=True, verbose=verbose
                    ).all()
                    or graph_full
                ):
                    self.get_progress_info(
                        search_step,
                        num_sub_goals,
                        print_percentage=True,
                        verbose=verbose,
                    )
                    break

        return graph_full


class ExplorerWorker(CommonGoalWorkerInterface):
    def work(self, verbose: bool = False) -> bool:
        """Tries to find paths to randomly sampled goals"""
        num_sub_goals = self.params.num_sub_goals
        for search_step in range(num_sub_goals + 1):
            if search_step == num_sub_goals:  # final goal
                self.graph.reset_sub_goal_states()
            else:
                new_sub_goal_states = sample_random_sub_goal_states(
                    self.parent.plant,
                    self.params,
                    size=self.params.num_parallel_searches,
                )
                self.graph.change_sub_goal_states(new_sub_goal_states)

            graph_full = self.parent.work(verbose=verbose)

            # do callbacks and termination checking here if we have sub goals
            if num_sub_goals > 0:
                if (
                    self.callback_and_progress_check(
                        search_step, num_sub_goals, change_goal=True, verbose=verbose
                    ).all()
                    or graph_full
                ):
                    self.get_progress_info(
                        search_step,
                        num_sub_goals,
                        print_percentage=True,
                        verbose=verbose,
                    )
                    break

        return graph_full


class RolloutWorker(GraphWorker):
    def work(self, verbose: bool = False) -> bool:
        """Always extends the last node."""
        params = self.params
        logger = self.logger

        for search_step in range(params.steps_per_goal):
            logger.log_search(search_step)

            node_ids = torch.tensor(
                [self.graph.next_main_node_id - 1], dtype=torch.int64
            )
            logger.log_node_selection(node_ids, SelectionType.LAST)

            relative_actions, num_action_steps, action_type = self.action_sampler(
                node_ids
            )
            logger.log_action_sampler(node_ids, action_type)
            node_ids, dynamics_time, graph_full = self.node_extension(
                node_ids, relative_actions, num_action_steps
            )
            logger.log_node_extension(
                node_ids, self.graph.get_best_id(reward_based=False), dynamics_time
            )

            # do callbacks here if we don't have sub goals
            if params.num_sub_goals == 0:
                self.callback_and_progress_check(
                    search_step, params.steps_per_goal, verbose=verbose
                ).all()
            if graph_full:
                break

        return graph_full


def inspect_action_type(
    graph_worker: GraphWorker,
    action_type: ActionType,
    node_ids: IntTensor | None = None,
    num_action_steps: int = 100,
) -> FloatTensor:
    """Inspection tool for a specific action type. This rollout the dynamics of the system
    assuming that we always select the same action_type.
    """
    params = graph_worker.params
    plant = graph_worker.plant
    graph = graph_worker.graph
    action_sampler = graph_worker.action_sampler

    # set the action type as the only possible action to sample
    params.action_types = [action_type]
    params.action_distribution = torch.ones(1)

    # initial state that we perform the rollout from
    if node_ids is None:
        node_ids = torch.tensor([0])

    state_trajectory = torch.zeros((0, plant.state_dimension))

    for _ in range(num_action_steps):
        # This ignores the num_action_steps
        relative_actions, _, sampled_action_type = action_sampler(node_ids)
        assert sampled_action_type == action_type

        states = graph.states[node_ids]
        start_actions = graph_worker.get_start_actions(node_ids)
        end_actions = graph_worker.get_end_actions(
            node_ids, relative_actions, action_type
        )
        start_end_sub_actions = torch.stack((start_actions, end_actions), dim=1)

        states, traj = plant.dynamics(
            states, start_end_sub_actions, params.action_time_step
        )
        state_trajectory = torch.cat((state_trajectory, traj[0, :, :]))

        node_ids, graph_full = graph.add_nodes(
            graph.root_ids[node_ids],
            node_ids,
            states,
            start_actions,
            end_actions,
            relative_actions,
            is_main_node=True,
        )
        if graph_full:
            break

    return state_trajectory
