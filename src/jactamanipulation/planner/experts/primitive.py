# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
"""PrimitiveSampler Submodule."""
import importlib
from typing import Dict, List

import numpy as np
import torch
from mujoco import rollout

from jactamanipulation.planner.dynamics.mujoco_dynamics import MujocoPlant
from jactamanipulation.planner.experts.expert_sampler import ExpertSampler
from jactamanipulation.planner.planner.graph import Graph
from jactamanipulation.planner.planner.parameter_container import ParameterContainer


class PrimitiveSampler(ExpertSampler):
    """PrimitiveSampler object.

    A 'primitive' is a higher-level action type that entails multiple
    low-level actuator actions to reach some subgoal.  It is meant to represent
    manipulation primitives such as "grasping", "pushing", etc. A
    PrimitiveSampler can compute the necessary low-level actions given an
    initial state, and iterative over individual actions in the proper sequence.
    """

    def _import_primitive_action_func(self, primitive: str) -> None:
        """Imports the specific action into the PrimitiveSampler.

        Primitive should be <module>.<function> where <module> corresponds to
        the file dexterity/primitives/<module>.py and <function> is a function
        which takes in a MujocoPlant, and returns a list of low-level
        actions, where each is represented by a tuple (start_action, end_action,
        action_time), used to create a PrimitiveSampler object.
        """
        module_name, function_name = primitive.split(".")
        module_full_path = f"dexterity.planner.experts.primitives.{module_name}"
        module = importlib.import_module(module_full_path)
        function = getattr(module, function_name)
        self.actions_func = function

    def __init__(
        self,
        plant: MujocoPlant,
        graph: Graph,
        params: ParameterContainer,
        actions_func: str,
        **actions_func_kwargs: Dict,
    ) -> None:
        """Constructs a PrimitiveSampler.

        Takes in a string actions_func: Given access to the simulator plant, this
        function returns a list of low-level actions, where each is represented by a
        tuple (start_action, end_action, action_time).
        """
        self.plant = plant
        self.graph = graph
        self.params = params
        self._import_primitive_action_func(actions_func)
        self.actions_func_kwargs = actions_func_kwargs if actions_func_kwargs is not None else {}

        # maps from node id to an action sequence originated from that node
        self.action_seqs: Dict[int, List] = {}
        # stores the node index responsible for starting the action sequence
        self.origins = torch.full((len(graph.states),), -1)
        # stores the step index of the ongoing action sequence
        self.ongoing = torch.full((len(graph.states),), -1)

    def callback(self, node_ids: torch.IntTensor) -> np.ndarray:
        """Given node_ids (N,), returns actions (N, na).

        Note that if node_ids contains duplicates, then we should
        be returning the same action for them, since they are the
        same nodes (representing the same stage in the primitive execution).
        """
        actions = []
        for nid in node_ids:
            nid = nid.item()
            parent = self.graph.parents[nid]

            if self.ongoing[nid] >= 0:
                # we have already computed an action for this node. will
                # return the same action.
                step_index = self.ongoing[nid]
                if self.ongoing[nid] == 0:
                    action_seq = self.action_seqs[nid]
                else:
                    action_seq = self.action_seqs[self.origins[parent].item()]
                actions.append(action_seq[step_index])
                continue

            recompute_actions = self.ongoing[parent] < 0
            if not recompute_actions:
                # we are in the middle of executing some action sequence
                action_seq = self.action_seqs[self.origins[parent].item()]
                step_index = self.ongoing[parent] + 1
                if step_index == len(action_seq):
                    # we are done. start over again.
                    recompute_actions = True

            if recompute_actions:
                # we have to recompute actions again (requires plant access)
                action_seq = self.compute_low_level_actions(self.graph.states[nid], nid)
                self.action_seqs[nid] = action_seq
                step_index = 0

            # update our data structures
            if step_index == 0:
                self.origins[nid] = nid
            else:
                self.origins[nid] = self.origins[parent]

            self.ongoing[nid] = step_index
            actions.append(action_seq[step_index])
        return torch.tensor(actions, dtype=torch.float32)

    def compute_low_level_actions(self, state: torch.FloatTensor, node_id: int) -> List:
        """Compute complete action trajectory.

        NOTE/TODO: action_time is currently ignored; whatever the action_sampler
        function computes as action_time will overwrite the returned values here.
        """
        # set the plant state
        if state.ndim == 1:
            state = state[np.newaxis, :]
            state = torch.hstack((torch.ones((1, 1)) * self.plant.data.time, state))
        else:
            num_envs = state.shape[0]
            state = torch.hstack((torch.ones((num_envs, 1)) * self.plant.data.time, state))
        rollout.rollout(self.plant.model, self.plant.data, state.cpu().numpy(), np.zeros(len(self.plant.data.ctrl)))
        actions = self.actions_func(self.plant, **self.actions_func_kwargs)

        # we will convert actions from the tuple representation to a sequence
        # (so we are ignoring the times)
        start_action_of_first_step = actions[0][0]
        action_seq = [start_action_of_first_step]
        for _, end_action, _ in actions:
            action_seq.append(end_action)
        return action_seq

    def reset(self) -> None:
        """Clear action_seqs, origins, and ongoing for new trajectory."""
        self.action_seqs.clear()
        self.origins = torch.full((len(self.graph.states),), -1)
        self.ongoing = torch.full((len(self.graph.states),), -1)
