# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from planner_setup import planner_setup

from jacta.planner.core.graph import Graph
from jacta.planner.core.parameter_container import ParameterContainer


def slow_get_best_id(graph: Graph, reward_based: bool = True) -> int:
    """Get the best id from the graph. This is the original implementation that uses sorting."""
    sorted_list = graph.sorted_progress_ids(reward_based)
    return sorted_list[0]


def test_get_best_id() -> None:
    params = ParameterContainer()
    params.parse_params("planar_hand", "test")
    planner = planner_setup(params, True)
    graph = planner.graph

    best_id = graph.get_best_id(reward_based=False)
    slow_best_id = slow_get_best_id(graph, reward_based=False)
    assert best_id == slow_best_id

    best_id = graph.get_best_id(reward_based=True)
    slow_best_id = slow_get_best_id(graph, reward_based=True)
    assert best_id == slow_best_id
