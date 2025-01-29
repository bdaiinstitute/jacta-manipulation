# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Tuple

import torch
from torch import FloatTensor, IntTensor

from jactamanipulation.planner.dynamics.mujoco_dynamics import MujocoPlant
from jactamanipulation.planner.experts.expert import Expert
from jactamanipulation.planner.planner.graph import Graph
from jactamanipulation.planner.planner.linear_algebra import (
    einsum_ij_kj_ki,
    einsum_ijk_ikl_ijl,
    einsum_ikj_ik_ij,
    einsum_ikj_ikl_ijl,
    einsum_jk_ikl_ijl,
    max_scaling,
    normalize_multiple,
    project_vectors_on_eigenspace,
)
from jactamanipulation.planner.planner.parameter_container import ParameterContainer
from jactamanipulation.planner.planner.types import ACTION_TYPE_DIRECTIONAL_MAP, ActionType


class ActionSampler:
    """ActionSampler"""

    def __init__(
        self,
        plant: MujocoPlant,
        graph: Graph,
        params: ParameterContainer,
    ):
        self.initialize(plant, graph, params)

    def reset(self) -> None:
        """Re initializes the action sampler"""
        self.initialize(self.plant, self.graph, self.params)

    def initialize(
        self,
        plant: MujocoPlant,
        graph: Graph,
        params: ParameterContainer,
    ) -> None:
        """Initializes the action sampler internal state

        Args:
            plant (MujocoPlant): Simulation plant
            graph (Graph): Graph
            params (ParameterContainer): Params
        """
        self.params = params
        self.plant = plant
        self.graph = graph
        if "action_experts" in params:
            self.expert = Expert(plant, graph, params)

    def random_directions(self, node_ids: IntTensor) -> FloatTensor:
        """Generate a random direction."""
        action_range = self.params.action_range
        directions = torch.rand(len(node_ids), len(action_range)) * (2 * action_range) - action_range
        return normalize_multiple(directions)

    def proximity_directions(self, node_ids: IntTensor) -> FloatTensor:
        """Generate a direction based on the proximity gradient."""
        graph = self.graph

        # TODO should use sub-stepped gradients, but slow at the moment
        # For einsum explanation, see https://ajcr.net/Basic-guide-to-einsum/
        sensor_gradients_control = einsum_ijk_ikl_ijl(
            graph.sensor_gradients_state[node_ids], graph.state_gradients_control_stepped[node_ids]
        )
        directions = -einsum_ikj_ik_ij(sensor_gradients_control, graph.sensors[node_ids])
        return normalize_multiple(directions)

    def continuation_directions(self, node_ids: IntTensor) -> FloatTensor:
        """Generate same direction as in-edge action."""
        graph = self.graph
        directions = graph.end_actions[node_ids] - graph.start_actions[node_ids]
        return normalize_multiple(directions)

    def goal_directions(self, node_ids: IntTensor) -> FloatTensor:
        """Generates directions with dynamics gradients.

        We formulate a quadratic objective from linearized dynamics, the action minimizing
        the distance to goal is calculated with optimization.
        """
        # TODO(slecleach) this ignores the non linearity of state addition and state difference
        plant = self.plant
        graph = self.graph
        params = self.params

        states = graph.states[node_ids]
        root_ids = graph.root_ids[node_ids]
        state_goal = graph.sub_goal_states[root_ids]
        Bs = graph.state_gradients_control_stepped[node_ids]

        # TODO(slecleach) here we simulate dynamics forward in time, this action should only be taken from a node
        # that already has a child to use their state instead of rolling forward the dynamics
        states_next_approx = states

        scaled_Bs = einsum_jk_ikl_ijl(params.reward_distance_scaling, Bs)
        regularization = (
            torch.tile(torch.eye(plant.action_dimension), (len(node_ids), 1, 1)) * params.action_regularization
        )
        goal_weights = einsum_ikj_ikl_ijl(Bs, scaled_Bs) + regularization
        state_differences = plant.state_difference(states_next_approx, state_goal)
        scaled_state_differences = einsum_ij_kj_ki(params.reward_distance_scaling, state_differences)
        mapped_state_differences = einsum_ikj_ik_ij(Bs, scaled_state_differences)

        try:
            directions = torch.linalg.solve(goal_weights, mapped_state_differences)
        except torch.linalg.LinAlgError:
            directions = torch.linalg.solve(regularization, mapped_state_differences)

        return normalize_multiple(directions)

    def directions_actions(self, node_ids: IntTensor, directions: FloatTensor) -> FloatTensor:
        """Calculate a set of actions based on sampled directions and the node's last action

        Args:
            node_ids: node ids we are looking to extend
            directions: (k, nq) set of directions are looking to expand the node in

        Returns:
            Set of trajectories expanded in the directions
        """
        params = self.params

        # Check if we are projecting on the eigenspace
        if params.using_eigenspaces:
            directions = project_vectors_on_eigenspace(directions, params.orthonormal_basis)
        max_scalings = max_scaling(directions, params.action_range * params.action_time_step)
        scalings = torch.rand(len(node_ids)) * max_scalings
        return directions * scalings.unsqueeze(1)

    def __call__(self, node_ids: IntTensor) -> Tuple[FloatTensor, int, ActionType]:
        """Combines each method of sampling to select an action and potentially project it into the eigenspace"""
        params = self.params
        action_idx = params.action_distribution.multinomial(num_samples=1, replacement=True)
        action_type = params.action_types[action_idx]
        num_action_steps = torch.randint(1, params.action_steps_max + 1, size=()).item()

        match action_type:
            case ActionType.RANGED:
                directions = self.random_directions(node_ids)
            case ActionType.PROXIMITY:
                directions = self.proximity_directions(node_ids)
            case ActionType.CONTINUATION:
                directions = self.continuation_directions(node_ids)
            case ActionType.GOAL:
                directions = self.goal_directions(node_ids)
            case ActionType.EXPERT:
                relative_actions = self.expert.expert_actions(node_ids)
            case _:
                raise NameError(f"Unknown action type {action_type}")

        if ACTION_TYPE_DIRECTIONAL_MAP[action_type]:
            relative_actions = self.directions_actions(node_ids, directions)

        return relative_actions, num_action_steps, action_type
