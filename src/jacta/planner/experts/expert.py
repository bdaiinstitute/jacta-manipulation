# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import importlib

from torch import FloatTensor, IntTensor

from jacta.planner.core.graph import Graph
from jacta.planner.core.parameter_container import ParameterContainer
from jacta.planner.dynamics.simulator_plant import SimulatorPlant


class Expert:
    def _import_experts(self) -> None:
        self.experts = []
        for i, expert in enumerate(self.params.action_experts):
            module_name, function_name = expert.split(".")
            module_full_path = f"jacta.planner.experts.{module_name}"
            module = importlib.import_module(module_full_path)
            expert_class = getattr(module, function_name)
            kwargs = eval(self.params.action_expert_kwargs[i])
            expert_obj = expert_class(self.plant, self.graph, self.params, **kwargs)
            self.experts.append(expert_obj)

    def __init__(
        self,
        plant: SimulatorPlant,
        graph: Graph,
        params: ParameterContainer,
    ):
        # Creates an expert sampler that contains a list of expert objects

        self.plant = plant
        self.graph = graph
        self.params = params
        self.distribution = params.action_expert_distribution
        self._import_experts()

    def expert_actions(self, node_ids: IntTensor) -> FloatTensor:
        expert_action_idx = self.distribution.multinomial(
            num_samples=1, replacement=True
        )
        actions = self.experts[expert_action_idx].callback(node_ids)
        return actions
