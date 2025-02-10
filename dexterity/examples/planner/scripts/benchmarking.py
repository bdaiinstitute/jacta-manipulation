# %%
# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import torch

from dexterity.jacta_planner.benchmarking.benchmarking import Benchmark
from dexterity.jacta_planner.dynamics.mujoco_dynamics import MujocoPlant
from dexterity.jacta_planner.planner.action_sampler import ActionSampler
from dexterity.jacta_planner.planner.graph import Graph
from dexterity.jacta_planner.planner.graph_worker import SingleGoalWorker
from dexterity.jacta_planner.planner.logger import Logger
from dexterity.jacta_planner.planner.parameter_container import ParameterContainer
from dexterity.jacta_planner.planner.planner import Planner

# %%
examples_list = [
    ["single_goal", "allegro_hand_eigenspace"],
    ["single_goal", "allegro_hand"],
    ["single_goal", "bimanual_kuka_allegro"],
    ["single_goal", "bimanual_station"],
    ["single_goal", "box_push"],
    ["single_goal", "cartpole"],
    ["single_goal", "floating_hand"],
    ["single_goal", "kuka_allegro"],
    ["single_goal", "planar_hand_eigenspace"],
    ["single_goal", "planar_hand"],
    ["single_goal", "satellite_offcenter"],
    ["single_goal", "satellite"],
    ["single_goal", "spot_bimanual_box"],
    ["single_goal", "spot_floating_box"],
    ["single_goal", "spot_standing_box"],
    ["multi_goal", "allegro_hand"],
    ["multi_goal", "bimanual_station"],
    ["multi_goal", "floating_hand"],
    ["multi_goal", "planar_hand"],
]

planners = []

for _, task in examples_list:
    params = ParameterContainer(f"dexterity/examples/jacta_planner/config/{task}.yml")

    plant = MujocoPlant(params=params)
    graph = Graph(plant, params)
    logger = Logger(graph, params)
    action_sampler = ActionSampler(plant, graph, params)
    graph_worker = SingleGoalWorker(plant, graph, action_sampler, logger, params)

    planner = Planner(plant, graph, action_sampler, graph_worker, logger, params, verbose=False)
    planners.append(planner)

seeds = [torch.arange(5) for _ in range(len(planners))]
names = [example[1] + "_" + example[0] for example in examples_list]
benchmark = Benchmark(planners, seeds, names, verbose=False)
benchmark.run_benchmark()
benchmark.print_results()
benchmark.visualize_results()
