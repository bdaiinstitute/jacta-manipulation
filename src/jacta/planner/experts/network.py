# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from pathlib import Path

import torch
from torch import FloatTensor, IntTensor

from jacta.learning.networks import Actor
from jacta.learning.normalizer import Normalizer
from jacta.planner.core.data_collection import find_latest_model_path, load_model
from jacta.planner.core.graph import Graph
from jacta.planner.core.parameter_container import ParameterContainer
from jacta.planner.dynamics.simulator_plant import SimulatorPlant
from jacta.planner.experts.expert_sampler import ExpertSampler


class NetworkSampler(ExpertSampler):
    def __init__(
        self,
        plant: SimulatorPlant,
        graph: Graph,
        params: ParameterContainer,
        path: str = "",
        model_name: str = "actor.pt",
        state_norm_name: str = "state_norm.pt",
    ):
        self.params = params
        self.plant = plant
        self.graph = graph

        size_s = plant.state_dimension
        size_a = plant.action_dimension
        self.actor = Actor(size_s * 2, size_a)
        self.state_norm = Normalizer(size_s)

        task = self.params.model_filename[:-4]

        if path:
            self.path = path
        else:
            base_path = str(Path(__file__).resolve().parent.parents[3])
            base_local_path = base_path + f"/examples/learning/models/{task}/"
            self.path = find_latest_model_path(base_local_path)

        load_model(self.actor, self.path + model_name)
        load_model(self.state_norm, self.path + state_norm_name)

    def callback(self, node_ids: IntTensor) -> FloatTensor:
        node_root_ids = self.graph.root_ids[node_ids]
        sub_goals = self.graph.sub_goal_states[node_root_ids]
        with torch.no_grad():
            actor_actions = self.actor.select_action(
                self.state_norm, self.graph.states[node_ids], sub_goals
            )
        return actor_actions * self.params.action_range * self.params.action_time_step
