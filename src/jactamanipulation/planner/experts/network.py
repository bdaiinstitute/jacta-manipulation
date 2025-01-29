# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
"""NetworkSampler Submodule."""
from pathlib import Path

import torch
from torch import FloatTensor, IntTensor

from dexterity.learning.networks import Actor
from dexterity.learning.normalizer import Normalizer
from dexterity.planner.dynamics.mujoco_dynamics import MujocoPlant
from dexterity.planner.experts.expert_sampler import ExpertSampler
from dexterity.planner.planner.data_collection import find_latest_model_path, load_model
from dexterity.planner.planner.graph import Graph
from dexterity.planner.planner.parameter_container import ParameterContainer


class NetworkSampler(ExpertSampler):
    """NetworkSampler object."""

    def __init__(
        self,
        plant: MujocoPlant,
        graph: Graph,
        params: ParameterContainer,
        is_local: bool = True,
        path: str = "",
        model_name: str = "actor.pt",
        state_norm_name: str = "state_norm.pt",
    ):
        """Constructs a NetworkSampler object."""
        self.params = params
        self.plant = plant
        self.graph = graph

        size_s = plant.state_dimension
        size_a = plant.action_dimension
        self.actor = Actor(size_s * 2, size_a)
        self.state_norm = Normalizer(size_s)

        task = self.params.model_filename[:-4]

        if path:
            self.path = Path(path)
        elif is_local:
            base_local_path = Path("dexterity/examples/learning/models") / task
            self.path = find_latest_model_path(base_local_path)
        else:
            base_cloud_path = Path("dexterity") / task / "models"
            self.path = find_latest_model_path(base_cloud_path)

        load_model(self.actor, self.path / model_name, is_local)
        load_model(self.state_norm, self.path / state_norm_name, is_local)

    def callback(self, node_ids: IntTensor) -> FloatTensor:
        """Query a neural net to sample actions."""
        node_root_ids = self.graph.root_ids[node_ids]
        sub_goals = self.graph.sub_goal_states[node_root_ids]
        states = self.state_norm(self.graph.states[node_ids])
        obs = torch.cat([states, sub_goals], dim=1)
        with torch.no_grad():
            actor_actions = self.actor.select_action(self.state_norm, obs)
        return actor_actions * self.params.action_range * self.params.action_time_step
