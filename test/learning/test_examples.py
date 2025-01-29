# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
import os

import torch
from benedict import benedict

from dexterity.learning.learner import Learner
from dexterity.learning.replay_buffer import ReplayBuffer
from dexterity.planner.dynamics.mujoco_dynamics import MujocoPlant
from dexterity.planner.planner.action_sampler import ActionSampler
from dexterity.planner.planner.graph import Graph, sample_random_goal_states
from dexterity.planner.planner.graph_worker import ExplorerWorker, RolloutWorker
from dexterity.planner.planner.logger import Logger
from dexterity.planner.planner.parameter_container import ParameterContainer
from dexterity.planner.planner.planner import Planner
from dexterity.planner.planner.types import ActionType as AT


def test_examples() -> None:
    examples_folders = ["learning"]

    for examples_folder in examples_folders:
        path_to_config = "dexterity/examples/" + examples_folder + "/config/"
        config_files = os.listdir(path_to_config)
        for config_file in config_files:
            params = ParameterContainer(f"dexterity/examples/{examples_folder}/config/{config_file}")

            base_values = {
                "steps_per_goal": 2,
                "num_sub_goals": 2,
                "action_time_step": 0.4,
                "action_steps_max": 2,
                "learner": benedict({"cycles": 2, "rollouts": 2, "batches": 2, "early_stop": 1.0}),
            }
            params.update(base_values)

            plant = MujocoPlant(params=params)
            graph = Graph(plant, params)
            graph.set_start_states(params.start_state.unsqueeze(0))
            logger = Logger(graph, params)
            action_sampler = ActionSampler(plant, graph, params)
            graph_worker = ExplorerWorker(plant, graph, action_sampler, logger, params)
            planner = Planner(plant, graph, action_sampler, graph_worker, logger, params, verbose=False)
            planner.search()

            replay_buffer = ReplayBuffer(params)
            learner = Learner(plant, graph, replay_buffer, params, verbose=False)
            learner.train(num_epochs=2)
            replay_buffer_0 = replay_buffer["states"].clone()

            planner.reset()
            graph.set_start_states(params.start_state.unsqueeze(0))
            planner.search()

            learner.reset()
            learner.train(num_epochs=2)
            replay_buffer_1 = replay_buffer["states"].clone()

            assert torch.any(replay_buffer_0 != 0)
            assert torch.any(replay_buffer_1 != 0)
            assert torch.allclose(replay_buffer_0, replay_buffer_1)

            rollout_values = {
                "action_types": [AT.EXPERT],
                "action_distribution": torch.ones(1),
                "action_experts": ["network.NetworkSampler"],
                "action_expert_kwargs": ["{}"],
                "action_expert_distribution": torch.ones(1),
                "steps_per_goal": "self.learner_trajectory_length * 2",
                "num_sub_goals": 0,
            }
            params.update(rollout_values)

            graph.reset()
            logger.reset()
            action_sampler.reset()
            graph_worker = RolloutWorker(plant, graph, action_sampler, logger, params)
            planner = Planner(plant, graph, action_sampler, graph_worker, logger, params, verbose=False)

            graph.change_sub_goal_states(sample_random_goal_states(plant, params))

            planner.search()

            assert True
