# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
import torch
from planner_setup import planner_setup

from jacta.planner.core.graph_worker import pareto_distribution
from jacta.planner.core.parameter_container import ParameterContainer


def test_pareto_distribution() -> None:
    params = ParameterContainer()
    params.parse_params("planar_hand", "test")
    planner_setup(params, False)
    vals = torch.tensor([0, 1, 2])
    shuffle_idx = torch.randperm(vals.shape[0])
    vals = vals[shuffle_idx]

    distribution = pareto_distribution(len(vals), 2.0)
    print(distribution)

    ids = vals[distribution.multinomial(num_samples=1000, replacement=True)]

    occurrences = torch.tensor([torch.count_nonzero(ids == idx) for idx in vals])
    print(occurrences)
    assert occurrences[0] > occurrences[1] > occurrences[2]


def test_pruning() -> None:
    params = ParameterContainer()
    params.parse_params("planar_hand", "test")
    planner = planner_setup(params, True)
    graph = planner.graph
    for node_id in graph.get_active_main_ids():
        if node_id not in graph.parents:  # just check leaves
            parent_id = node_id
            while parent_id != 0:
                parent_id = graph.parents[parent_id]
                if graph.main_nodes[parent_id]:  # dont check sub nodes
                    assert graph.is_better_than(node_id, parent_id)
                    break


def test_parallel_search_exploration() -> None:
    params = ParameterContainer()
    params.parse_params("planar_hand", "test")
    params.num_parallel_searches = 3
    params.intermediate_pruning = False
    params.intermediate_replacement = False
    planner_setup(params, True)
