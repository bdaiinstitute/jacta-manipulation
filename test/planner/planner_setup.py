# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from jacta.planner.core.action_sampler import ActionSampler
from jacta.planner.core.graph import Graph
from jacta.planner.core.graph_worker import ParallelGoalsWorker, SingleGoalWorker
from jacta.planner.core.logger import Logger
from jacta.planner.core.parameter_container import ParameterContainer
from jacta.planner.core.planner import Planner
from jacta.planner.dynamics.mujoco_dynamics import MujocoPlant


def planner_setup(params: ParameterContainer, search: bool) -> Planner:
    plant = MujocoPlant(params)
    graph = Graph(plant, params)
    logger = Logger(graph, params)
    action_sampler = ActionSampler(plant, graph, params)
    if params.num_parallel_searches > 1:
        graph_worker_class = ParallelGoalsWorker
    else:
        graph_worker_class = SingleGoalWorker
    graph_worker = graph_worker_class(plant, graph, action_sampler, logger, params)

    planner = Planner(
        plant, graph, action_sampler, graph_worker, logger, params, verbose=False
    )

    if search:
        params.steps_per_goal = 100
        planner.search()

    return planner
