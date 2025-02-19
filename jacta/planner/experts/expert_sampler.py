# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import torch
from torch import FloatTensor, IntTensor

from jacta.planner.dynamics.simulator_plant import SimulatorPlant
from jacta.planner.planner.graph import Graph
from jacta.planner.planner.linear_algebra import normalize_multiple
from jacta.planner.planner.parameter_container import ParameterContainer


class ExpertSampler:
    def __init__(self, plant: SimulatorPlant, graph: Graph, params: ParameterContainer):
        self.params = params
        self.plant = plant
        self.graph = graph

    def callback(self, node_ids: IntTensor) -> FloatTensor:
        # Default callback: random action generation
        action_range = self.params.action_range
        directions = torch.rand(len(node_ids), len(action_range)) * (2 * action_range) - action_range
        return normalize_multiple(directions)
