# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from pathlib import Path

import torch
from benedict import benedict

from jacta.planner.core.action_sampler import ActionSampler
from jacta.planner.core.graph import Graph
from jacta.planner.core.graph_worker import (
    ExplorerWorker,
    RelatedGoalWorker,
    SingleGoalWorker,
)
from jacta.planner.core.logger import Logger
from jacta.planner.core.parameter_container import ParameterContainer
from jacta.planner.core.planner import Planner
from jacta.planner.dynamics.mujoco_dynamics import MujocoPlant


def get_planner_examples(config_path: str) -> list[str]:
    config_dict = benedict.from_yaml(config_path)
    examples = config_dict.get_dict("planner").keys()
    if len(examples) == 0:
        return ["single_goal"]
    else:
        return examples


def test_examples() -> None:
    base_path = str(Path(__file__).resolve().parent.parents[1])
    path_to_config = base_path + "/examples/planner/config/task/"
    config_files = os.listdir(path_to_config)
    for task in config_files:
        for example in get_planner_examples(path_to_config + task):
            print(f"Task: {task}, example: {example}")
            params = ParameterContainer()
            params.parse_params(task[:-4], example)
            params.steps_per_goal = 2
            if params.num_parallel_searches > 1:
                print("Skipping test for parallel searches")
                continue

            plant = MujocoPlant(params)
            graph = Graph(plant, params)
            logger = Logger(graph, params)
            action_sampler = ActionSampler(plant, graph, params)
            match example:
                case "single_goal":
                    graph_worker = SingleGoalWorker(
                        plant, graph, action_sampler, logger, params
                    )
                case "multi_goal":
                    graph_worker = RelatedGoalWorker(
                        plant, graph, action_sampler, logger, params
                    )
                    params.num_sub_goals = 2
                case "exploration":
                    graph_worker = ExplorerWorker(
                        plant, graph, action_sampler, logger, params
                    )
                    params.num_sub_goals = 2
            planner = Planner(
                plant,
                graph,
                action_sampler,
                graph_worker,
                logger,
                params,
                verbose=False,
            )

            planner.search()
            search_index = 0
            trajectory_0 = planner.shortest_path_trajectory(search_index=search_index)
            states_0 = planner.graph.states

            planner.reset()

            planner.search()
            trajectory_1 = planner.shortest_path_trajectory(search_index=search_index)
            states_1 = planner.graph.states

            assert torch.allclose(states_0, states_1)
            assert trajectory_0.shape == trajectory_1.shape
            assert torch.allclose(trajectory_0, trajectory_1)
