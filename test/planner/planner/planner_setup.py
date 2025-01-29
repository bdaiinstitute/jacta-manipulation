# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from jactamanipulation.planner.dynamics.mujoco_dynamics import MujocoPlant
from jactamanipulation.planner.planner.action_sampler import ActionSampler
from jactamanipulation.planner.planner.graph import Graph
from jactamanipulation.planner.planner.graph_worker import ParallelGoalsWorker, SingleGoalWorker
from jactamanipulation.planner.planner.logger import Logger
from jactamanipulation.planner.planner.parameter_container import ParameterContainer
from jactamanipulation.planner.planner.planner import Planner


def planner_setup(params: ParameterContainer, search: bool) -> Planner:
    plant = MujocoPlant(params=params)
    graph = Graph(plant, params)
    logger = Logger(graph, params)
    action_sampler = ActionSampler(plant, graph, params)
    if params.num_parallel_searches > 1:
        graph_worker_class = ParallelGoalsWorker
    else:
        graph_worker_class = SingleGoalWorker
    graph_worker = graph_worker_class(plant, graph, action_sampler, logger, params)

    planner = Planner(plant, graph, action_sampler, graph_worker, logger, params, verbose=False)

    if search:
        params.steps_per_goal = 100
        planner.search()

    return planner
