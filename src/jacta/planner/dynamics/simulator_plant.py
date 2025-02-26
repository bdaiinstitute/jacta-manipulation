# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import torch
from torch import FloatTensor

from jacta.planner.core.parameter_container import ParameterContainer


class SimulatorPlant:
    def __init__(self, params: ParameterContainer):
        self.params = params

    def state_difference(
        self, s1: FloatTensor, s2: FloatTensor, h: float = 1.0
    ) -> FloatTensor:
        return torch.zeros_like(s1)  # To avoid not_implemented error


def scaled_distances_to(
    plant: SimulatorPlant, states: FloatTensor, target_states: FloatTensor
) -> FloatTensor:
    delta_states = plant.state_difference(states, target_states)
    return scale_distances(delta_states, plant.params.reward_distance_scaling_sqrt)


def scale_distances(delta_states: FloatTensor, scaling: FloatTensor) -> FloatTensor:
    return torch.norm(delta_states @ scaling, dim=-1)
