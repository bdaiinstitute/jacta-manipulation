# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from dexterity.planner.dynamics.mujoco_dynamics import MujocoPlant
from dexterity.planner.planner.action_sampler import ActionSampler
from dexterity.planner.planner.graph import Graph
from dexterity.planner.planner.graph_worker import ParallelGoalsWorker, SingleGoalWorker
from dexterity.planner.planner.logger import Logger
from dexterity.planner.planner.parameter_container import ParameterContainer
from dexterity.planner.planner.planner import Planner


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
