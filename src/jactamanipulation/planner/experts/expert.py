# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
"""Expert Submodule."""
import importlib

from torch import FloatTensor, IntTensor

from jactamanipulation.planner.dynamics.mujoco_dynamics import MujocoPlant
from jactamanipulation.planner.planner.graph import Graph
from jactamanipulation.planner.planner.parameter_container import ParameterContainer


class Expert:
    """Expert object."""

    def _import_experts(self) -> None:
        """Import all ExpertSampler's into the Expert class.

        Primitive should be <module>.<function> where <module> corresponds to
        the file dexterity/primitives/<module>.py and <function> is a function
        which takes in a MujocoPlant, and returns a list of low-level
        actions, where each is represented by a tuple (start_action, end_action,
        action_time), used to create a PrimitiveSampler object.
        """
        self.experts = []
        for i, expert in enumerate(self.params.action_experts):
            module_name, function_name = expert.split(".")
            module_full_path = f"dexterity.planner.experts.{module_name}"
            module = importlib.import_module(module_full_path)
            expert_class = getattr(module, function_name)
            kwargs = eval(self.params.action_expert_kwargs[i])
            expert_obj = expert_class(self.plant, self.graph, self.params, **kwargs)
            self.experts.append(expert_obj)

    def __init__(
        self,
        plant: MujocoPlant,
        graph: Graph,
        params: ParameterContainer,
    ):
        """Creates an expert that contains a list of ExpertSamplers."""
        self.plant = plant
        self.graph = graph
        self.params = params
        self.distribution = params.action_expert_distribution
        self._import_experts()

    def expert_actions(self, node_ids: IntTensor) -> FloatTensor:
        """Selects an ExpertSampler according to expert distribtuion to sample actions."""
        expert_action_idx = self.distribution.multinomial(num_samples=1, replacement=True)
        actions = self.experts[expert_action_idx].callback(node_ids)
        return actions
