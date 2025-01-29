# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from pathlib import Path

import torch
from benedict import benedict

from jactamanipulation.planner.dynamics.mujoco_dynamics import MujocoPlant
from jactamanipulation.planner.planner.action_processor import SpotWholebodyActionProcessor
from jactamanipulation.planner.planner.action_sampler import ActionSampler
from jactamanipulation.planner.planner.graph import Graph
from jactamanipulation.planner.planner.graph_worker import ExplorerWorker, RelatedGoalWorker, SingleGoalWorker
from jactamanipulation.planner.planner.logger import Logger
from jactamanipulation.planner.planner.parameter_container import ParameterContainer
from jactamanipulation.planner.planner.planner import Planner


def get_examples(config_path: Path) -> list[str]:
    config_dict = benedict.from_yaml(config_path)
    examples = config_dict.get_dict("planner").keys()
    if len(examples) == 0:
        return ["single_goal"]
    else:
        return examples


def test_examples() -> None:
    path_to_config = Path("dexterity/examples/planner/config/")
    config_files = os.listdir(path_to_config)
    for task in config_files:
        if task == "base.yml":
            continue
        for example in get_examples(path_to_config / task):
            print(f"Task: {task}, example: {example}")
            params = ParameterContainer(path_to_config / task)
            params.steps_per_goal = 2
            params.set_seed(42)
            if params.num_parallel_searches > 1:
                print("Skipping test for parallel searches")
                # TODO(DMM-2174):parallel search is currently not deterministic
                continue

            if params.action_processor_class is SpotWholebodyActionProcessor:
                print("Skipping test for SpotWholebodyActionProcessor")
                # TODO(maks): remove once the graph search is implemented
                continue

            plant = MujocoPlant(params=params)
            graph = Graph(plant, params)
            logger = Logger(graph, params)
            action_sampler = ActionSampler(plant, graph, params)
            match example:
                case "single_goal" | "primitives":
                    graph_worker = SingleGoalWorker(plant, graph, action_sampler, logger, params)
                case "multi_goal":
                    graph_worker = RelatedGoalWorker(plant, graph, action_sampler, logger, params)
                    params.num_sub_goals = 2
                case "exploration":
                    graph_worker = ExplorerWorker(plant, graph, action_sampler, logger, params)
                    params.num_sub_goals = 2
            planner = Planner(plant, graph, action_sampler, graph_worker, logger, params, verbose=False)

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
