# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
"""Expert Sampler Submodule."""
import torch
from torch import FloatTensor, IntTensor

from jactamanipulation.planner.dynamics.mujoco_dynamics import MujocoPlant
from jactamanipulation.planner.planner.graph import Graph
from jactamanipulation.planner.planner.linear_algebra import normalize_multiple
from jactamanipulation.planner.planner.parameter_container import ParameterContainer


class ExpertSampler:
    """ExpertSampler object."""

    def __init__(self, plant: MujocoPlant, graph: Graph, params: ParameterContainer):
        """Constructs an ExpertSampler object."""
        self.params = params
        self.plant = plant
        self.graph = graph

    def callback(self, node_ids: IntTensor) -> FloatTensor:
        """Default action callback of random action generation."""
        action_range = self.params.action_range
        directions = torch.rand(len(node_ids), len(action_range)) * (2 * action_range) - action_range
        return normalize_multiple(directions)
