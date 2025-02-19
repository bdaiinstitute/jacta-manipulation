# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import warnings
from typing import Optional, Tuple, Union

import torch
from torch import FloatTensor, IntTensor

from jacta.planner.core.linear_algebra import (
    einsum_ij_ij_i,
    einsum_ijk_ik_ij,
    einsum_ijk_ikl_ijl,
    einsum_ijk_ilk_ijl,
)
from jacta.planner.core.parameter_container import ParameterContainer
from jacta.planner.dynamics.simulator_plant import SimulatorPlant, scale_distances


def sample_related_sub_goal_states(
    params: ParameterContainer,
    goal_states: FloatTensor,
    start_states: FloatTensor,
    size: int = 1,
) -> FloatTensor:
    """
    Generates related goal states based on the provided parameters.
    Args:
        params (ParameterContainer): The container holding parameters of sub goal bounds.
        goal_states (FloatTensor): The goal states to generate related goal states from.
        start_states (FloatTensor): The start states to generate related goal states from.
        size (int, optional): Number of goal states to generate. Defaults to 1.
    Returns:
        Related goal states as a tensor.
    Note:
        This function assumes a diagonal covariance matrix.
        It relies on the fact that the entries are independent and identically distributed (i.i.d.) entries.
    """
    state_dimension = goal_states.shape[1]
    search_std = torch.abs(goal_states - start_states) / 2
    new_goal_states = torch.randn((size, state_dimension)) * search_std + goal_states
    return torch.clamp(
        new_goal_states,
        min=params.goal_sub_bound_lower,
        max=params.goal_sub_bound_upper,
    )


def sample_feasible_states(
    plant: SimulatorPlant,
    bound_lower: FloatTensor,
    bound_upper: FloatTensor,
    size: int = 1,
) -> FloatTensor:
    # Given 5 attempts to protect against potentially unavoidable collisions and getting stuck
    for _ in range(5):
        random_states = sample_random_states(bound_lower, bound_upper, size)
        normalized_states = plant.normalize_state(random_states)
        if plant.params.ignore_sampled_state_collisions:
            return normalized_states
        feasible_states = plant.get_collision_free(normalized_states)
        if feasible_states is not None:
            return feasible_states
    warnings.warn(
        "No collision free states could be sampled.", RuntimeWarning, stacklevel=2
    )
    return normalized_states


def sample_random_states(
    bound_lower: FloatTensor, bound_upper: FloatTensor, size: int = 1
) -> FloatTensor:
    return (
        torch.rand(size=(size, len(bound_upper))) * (bound_upper - bound_lower)
        + bound_lower
    )


def sample_random_start_states(
    plant: SimulatorPlant, params: ParameterContainer, size: int = 1
) -> FloatTensor:
    return sample_feasible_states(
        plant, params.start_bound_lower, params.start_bound_upper, size=size
    )


def sample_random_goal_states(
    plant: SimulatorPlant, params: ParameterContainer, size: int = 1
) -> FloatTensor:
    return sample_feasible_states(
        plant, params.goal_bound_lower, params.goal_bound_upper, size=size
    )


def sample_random_sub_goal_states(
    plant: SimulatorPlant, params: ParameterContainer, size: int = 1
) -> FloatTensor:
    return sample_feasible_states(
        plant, params.goal_sub_bound_lower, params.goal_sub_bound_upper, size=size
    )


class Graph:
    def __init__(self, plant: SimulatorPlant, params: ParameterContainer):
        self.plant = plant
        self.params = params
        self._initialize()

    def reset(self) -> None:
        """Fully resets the graph data for a new search."""
        self._initialize()

    def _initialize(self) -> None:
        """Initializes the graph data structures."""
        plant = self.plant
        params = self.params

        max_main_nodes = params.max_main_nodes
        max_all_nodes = params.max_main_nodes * params.action_steps_max

        # Create all data containers
        # required for all nodes
        self.ids = torch.arange(max_all_nodes)
        self.active_ids = torch.zeros(max_all_nodes, dtype=torch.bool)
        self.filled_ids = torch.zeros(max_all_nodes, dtype=torch.bool)
        self.main_nodes = torch.zeros(
            max_all_nodes, dtype=torch.bool
        )  # indicates whether main node or sub node
        self.states = torch.zeros((max_all_nodes, plant.state_dimension))
        self.start_actions = torch.zeros((max_all_nodes, plant.action_dimension))
        self.end_actions = torch.zeros((max_all_nodes, plant.action_dimension))
        self.relative_actions = torch.zeros(
            (max_all_nodes, plant.action_dimension)
        )  # without clipping to absolute bounds
        self.parents = torch.zeros(max_all_nodes, dtype=torch.int64)
        self.root_ids = torch.full(
            (max_all_nodes,), -1, dtype=torch.int64
        )  # overloaded to be search index

        # only required for main nodes
        self.scaled_goal_distances = torch.zeros(max_main_nodes)
        self.sensors = torch.zeros((max_main_nodes, plant.sensor_dimension))
        self.state_gradients_state = torch.zeros(
            (
                max_main_nodes,
                plant.state_derivative_dimension,
                plant.state_derivative_dimension,
            )
        )
        self.state_gradients_control = torch.zeros(
            (max_main_nodes, plant.state_derivative_dimension, plant.action_dimension)
        )
        self.sensor_gradients_state = torch.zeros(
            (max_main_nodes, plant.sensor_dimension, plant.state_derivative_dimension)
        )
        self.sensor_gradients_control = torch.zeros(
            (max_main_nodes, plant.sensor_dimension, plant.action_dimension)
        )
        self.state_gradients_control_stepped = torch.zeros(
            (max_main_nodes, plant.state_derivative_dimension, plant.action_dimension)
        )
        self.reachability_matrices = torch.zeros(
            (
                max_main_nodes,
                plant.unactuated_pos_difference.shape[0],
                plant.unactuated_pos_difference.shape[0],
            )
        )
        self.distance_rewards = torch.zeros(max_main_nodes)
        self.proximity_rewards = torch.zeros(max_main_nodes)
        self.reachability_rewards = torch.zeros(max_main_nodes)
        self.rewards = torch.zeros(max_main_nodes)

        self._reset_id_pointers()
        self._initialize_start_goal_states()
        self._initialize_root_nodes()

    def _reset_id_pointers(self) -> None:
        self.next_main_node_id = 0
        self.next_sub_node_id = self.params.max_main_nodes

    def _initialize_root_nodes(self) -> None:
        plant = self.plant
        params = self.params
        root_node_ids = torch.arange(params.num_parallel_searches)
        self.root_ids[: params.num_parallel_searches] = root_node_ids
        self.search_indices = torch.arange(params.num_parallel_searches)
        root_start_actions = self.start_states[:, plant.actuated_pos]
        root_end_actions = self.start_states[:, plant.actuated_pos]
        root_relative_actions = torch.zeros(
            (params.num_parallel_searches, plant.action_dimension)
        )
        self.add_nodes(
            root_ids=root_node_ids,
            parent_ids=root_node_ids,
            states=self.start_states,
            start_actions=root_start_actions,
            end_actions=root_end_actions,
            relative_actions=root_relative_actions,
        )

    def _initialize_start_goal_states(self) -> None:
        """Initializes the start, goal, and sub goal states."""
        self.start_states = sample_random_start_states(
            self.plant, self.params, size=self.params.num_parallel_searches
        )
        self.goal_states = sample_random_goal_states(
            self.plant, self.params, size=self.params.num_parallel_searches
        )
        self.sub_goal_states = self.goal_states.clone()

    def set_start_states(self, start_states: FloatTensor) -> None:
        self.start_states = start_states
        # reset the root nodes
        assert (
            self.next_main_node_id == self.params.num_parallel_searches
        ), "only able to set states right after graph initialization."
        self._reset_id_pointers()
        self._initialize_root_nodes()

    def set_goal_state(self, goal_state: FloatTensor) -> None:
        self.goal_states = goal_state.unsqueeze(0).repeat(
            self.params.num_parallel_searches
        )

    @property
    def node_id_to_search_index_map(self) -> IntTensor:
        return self.root_ids

    def calculate_distance_rewards(self, ids: IntTensor) -> FloatTensor:
        return -self.scaled_goal_distances[ids]

    def calculate_proximity_rewards(self, ids: IntTensor) -> FloatTensor:
        proximity_scaling = self.params.reward_proximity_scaling

        if proximity_scaling.ndim == 0:
            rewards = -torch.norm(
                torch.matmul(proximity_scaling, self.sensors[ids].T).T, dim=-1
            )
        else:
            rewards = -torch.norm(proximity_scaling * self.sensors[ids], dim=-1)
        return rewards

    def calculate_reachability_rewards(
        self,
        ids: IntTensor,
        delta_states: FloatTensor,
        minimum_distance: float = 0.001,
    ) -> FloatTensor:
        deltas = delta_states[..., self.plant.unactuated_pos_difference]
        scaled_deltas = einsum_ijk_ik_ij(self.reachability_matrices[ids], deltas)
        scaled_distances = einsum_ij_ij_i(deltas, scaled_deltas)
        limited_distances = torch.max(
            torch.vstack(
                (torch.ones_like(scaled_distances) * minimum_distance, scaled_distances)
            ),
            dim=0,
        ).values
        return -self.params.reward_reachability_scaling * (
            torch.log(limited_distances) - torch.log(torch.tensor([minimum_distance]))
        )

    def add_total_rewards(self, ids: IntTensor) -> FloatTensor:
        return (
            self.distance_rewards[ids]
            + self.proximity_rewards[ids]
            + self.reachability_rewards[ids]
        )

    def reachability_cache(self, ids: IntTensor) -> Tuple[FloatTensor, FloatTensor]:
        plant = self.plant
        num_substeps = plant.get_num_substeps(self.params.action_time_step)
        state_gradients_state = self.state_gradients_state[ids]
        state_gradients_control = self.state_gradients_control[ids]

        Bs = torch.zeros_like(state_gradients_control)
        for _ in range(num_substeps):
            Bs = state_gradients_control + einsum_ijk_ikl_ijl(state_gradients_state, Bs)
        Bus = Bs[:, plant.unactuated_pos_difference]

        if torch.isnan(Bs).any():
            warnings.warn(
                "graph.reachability_cache: Bs contains NaNs",
                RuntimeWarning,
                stacklevel=2,
            )

        regularization = (
            torch.tile(torch.eye(Bus.shape[1]), (len(ids), 1, 1))
            * self.params.reachability_regularization
        )
        inv_reachability_matrices = einsum_ijk_ilk_ijl(Bus, Bus) + regularization
        try:
            reachability_matrices = torch.linalg.inv(inv_reachability_matrices)
        except torch.linalg.LinAlgError:
            reachability_matrices = torch.linalg.inv(regularization)
        else:
            if torch.isnan(reachability_matrices).any():
                reachability_matrices = torch.linalg.inv(regularization)

        return Bs, reachability_matrices

    def add_nodes(
        self,
        root_ids: IntTensor,
        parent_ids: IntTensor,
        states: FloatTensor,
        start_actions: FloatTensor,
        end_actions: FloatTensor,
        relative_actions: FloatTensor,
        is_main_node: bool = True,
    ) -> Tuple[int, bool]:
        """Adds a new node to the graph based on its state/distance from the goal and updates its reward.

        When a new node is added to the graph, it gets evaluated in terms of reward and added to the graph.

        Args:
            parent_id: id to which the new node will be connected
            state: the current state of the node, used to determine its distance to goal
            action: the action used to reach the node

        Returns:
            The new ids and a flag if the graph is full.
        """
        if is_main_node:
            ids = torch.arange(
                self.next_main_node_id, self.next_main_node_id + len(parent_ids)
            )
            if ids[-1] >= self.params.max_main_nodes:
                warnings.warn(
                    "Graph is full, use a larger max_main_nodes parameter. Stopping search.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return parent_ids, True
            self.next_main_node_id = ids[-1].item() + 1
        else:
            ids = torch.arange(
                self.next_sub_node_id, self.next_sub_node_id + len(parent_ids)
            )
            if ids[-1] >= self.params.max_main_nodes * self.params.action_steps_max:
                warnings.warn(
                    "Graph is full, use a larger max_main_nodes parameter. Stopping search.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return parent_ids, True
            self.next_sub_node_id = ids[-1].item() + 1

        self.filled_ids[ids] = True
        self.active_ids[ids] = True
        self.main_nodes[ids] = is_main_node
        self.root_ids[ids] = root_ids
        self.parents[ids] = parent_ids
        self.states[ids] = states
        self.start_actions[ids] = start_actions
        self.end_actions[ids] = end_actions
        self.relative_actions[ids] = relative_actions

        if is_main_node:
            self._update_dynamics_info(ids, states, end_actions)
            self._update_rewards(ids)

        return ids, False

    def reset_sub_goal_states(self) -> None:
        """Resets the sub goal states to the goal states."""
        self.change_sub_goal_states(self.goal_states.clone())

    def change_sub_goal_states(
        self,
        sub_goal_states: FloatTensor,
    ) -> None:
        self.sub_goal_states = sub_goal_states
        ids = self.get_active_main_ids()
        self._update_rewards(ids)

    def _update_dynamics_info(
        self, ids: IntTensor, states: FloatTensor, end_actions: FloatTensor
    ) -> None:
        self.sensors[ids, :] = self.plant.get_sensor(states)
        (
            self.state_gradients_state[ids],
            self.state_gradients_control[ids],
            self.sensor_gradients_state[ids],
            self.sensor_gradients_control[ids],
        ) = self.plant.get_gradients(states, end_actions)
        self.state_gradients_control_stepped[ids], self.reachability_matrices[ids] = (
            self.reachability_cache(ids)
        )

    def _update_rewards(self, ids: IntTensor) -> None:
        search_indices = self.node_id_to_search_index_map[ids]
        delta_states = self.plant.state_difference(
            self.states[ids], self.sub_goal_states[search_indices]
        )
        self.scaled_goal_distances[ids] = scale_distances(
            delta_states, self.params.reward_distance_scaling_sqrt
        )
        self.distance_rewards[ids] = self.calculate_distance_rewards(ids)
        self.proximity_rewards[ids] = self.calculate_proximity_rewards(ids)
        self.reachability_rewards[ids] = self.calculate_reachability_rewards(
            ids, delta_states
        )
        self.rewards[ids] = self.add_total_rewards(ids)

    def deactivate_nodes(self, ids: IntTensor) -> None:
        self.active_ids[ids] = False

    def activate_all_nodes(self) -> None:
        """Converts all sub nodes to main nodes and activates all inactive but used nodes"""
        # extend data containers
        max_all_nodes = self.params.max_main_nodes * self.params.action_steps_max
        self.scaled_goal_distances.resize_(max_all_nodes)
        self.sensors.resize_((max_all_nodes, *self.sensors.shape[1:]))
        self.state_gradients_state.resize_(
            (max_all_nodes, *self.state_gradients_state.shape[1:])
        )
        self.state_gradients_control.resize_(
            (max_all_nodes, *self.state_gradients_control.shape[1:])
        )
        self.sensor_gradients_state.resize_(
            (max_all_nodes, *self.sensor_gradients_state.shape[1:])
        )
        self.sensor_gradients_control.resize_(
            (max_all_nodes, *self.sensor_gradients_control.shape[1:])
        )
        self.state_gradients_control_stepped.resize_(
            (max_all_nodes, *self.state_gradients_control_stepped.shape[1:])
        )
        self.reachability_matrices.resize_(
            (max_all_nodes, *self.reachability_matrices.shape[1:])
        )
        self.distance_rewards.resize_(max_all_nodes)
        self.proximity_rewards.resize_(max_all_nodes)
        self.reachability_rewards.resize_(max_all_nodes)
        self.rewards.resize_(max_all_nodes)

        # add data for sub nodes and convert to main nodes
        sub_node_ids = torch.nonzero(self.filled_ids * ~self.main_nodes).flatten()
        self._update_dynamics_info(
            sub_node_ids, self.states[sub_node_ids], self.end_actions[sub_node_ids]
        )
        self.main_nodes[sub_node_ids] = True

        # activate all inactive but used nodes
        inactive_ids = self.filled_ids * ~self.active_ids
        self.active_ids[inactive_ids] = True

        # update rewards
        self.change_sub_goal_states(self.sub_goal_states)

    def sorted_progress_ids(
        self, reward_based: bool, search_index: int = 0
    ) -> IntTensor:
        valid_ids = self.get_active_main_ids(search_index)
        if reward_based:
            rewards = self.rewards[valid_ids]
            sorted_indices = rewards.argsort(dim=-1, descending=True)
        else:
            distances = self.scaled_goal_distances[valid_ids]
            sorted_indices = distances.argsort(dim=-1, descending=False)
        return valid_ids[sorted_indices]

    def get_best_id(
        self, reward_based: bool = True, search_indices: Optional[IntTensor] = None
    ) -> int:
        if search_indices is None:
            search_indices = self.search_indices

        best_ids = torch.zeros_like(search_indices)
        for i, search_index in enumerate(search_indices):
            valid_ids = self.get_active_main_ids(search_index)
            if reward_based:
                rewards = self.rewards[valid_ids]
                best_ids[i] = valid_ids[rewards.argmax()]
            else:
                distances = self.scaled_goal_distances[valid_ids]
                best_ids[i] = valid_ids[distances.argmin()]
        return best_ids

    def is_worse_than(
        self, ids: Union[int, IntTensor], comparison_ids: int
    ) -> Union[bool, torch.Tensor]:
        if self.params.reward_based:
            return self.rewards[ids] < self.rewards[comparison_ids]
        else:
            return (
                self.scaled_goal_distances[ids]
                > self.scaled_goal_distances[comparison_ids]
            )

    def is_better_than(
        self, ids: Union[int, IntTensor], comparison_ids: int
    ) -> Union[bool, torch.Tensor]:
        if self.params.reward_based:
            return self.rewards[ids] > self.rewards[comparison_ids]
        else:
            return (
                self.scaled_goal_distances[ids]
                < self.scaled_goal_distances[comparison_ids]
            )

    def number_of_nodes(self) -> int:
        return len(self.get_active_main_ids())

    def get_active_main_ids(self, search_index: Optional[int] = None) -> IntTensor:
        valid_ids = self.ids[self.active_ids * self.main_nodes]
        if search_index is None:
            return valid_ids
        else:
            search_indices = self.node_id_to_search_index_map[valid_ids]
            return valid_ids[search_indices == search_index]

    def get_root_ids(self) -> IntTensor:
        valid_ids = self.get_active_main_ids()
        return torch.unique(self.root_ids[valid_ids])

    def shortest_path_to(self, idx: int, start_id: Optional[int] = None) -> IntTensor:
        if start_id is None:
            start_id = self.root_ids[idx]
        path = torch.zeros(0, dtype=torch.int)
        current_id = idx

        while True:
            path = torch.cat((torch.tensor([current_id]), path))

            parent_id = self.parents[current_id]
            if current_id == start_id:
                break
            elif current_id == parent_id:
                print("No path exists")
                break
            current_id = parent_id

        return path

    def save(self, filename: str, mask: IntTensor = slice(None)) -> None:  # noqa: B008
        data = {
            "states": self.states[mask],
            "start_actions": self.start_actions[mask],
            "end_actions": self.end_actions[mask],
            "relative_actions": self.relative_actions[mask],
            "parents": self.parents[mask],
            "root_ids": self.root_ids[mask],
            "scaled_goal_distances": self.scaled_goal_distances[mask],
            "sensors": self.sensors[mask],
            "distance_rewards": self.distance_rewards[mask],
            "proximity_rewards": self.proximity_rewards[mask],
            "reachability_rewards": self.reachability_rewards[mask],
            "rewards": self.rewards[mask],
            "ids": self.ids[mask],
            "filled_ids": self.filled_ids[mask],
            "active_ids": self.active_ids[mask],
            "main_nodes": self.main_nodes[mask],
        }

        if hasattr(self, "children"):
            data["children"] = self.children[mask]
            data["num_children"] = self.num_children[mask]

        torch.save(data, filename)

    def load(self, filename: str) -> None:
        data = torch.load(filename)
        for key, value in data.items():
            setattr(self, key, value)

    def add_child_ids_to_node(self) -> None:
        children: dict = {
            node_id.item(): [] for node_id in self.ids
        }  # {parent_id: [child_id]}
        for child_id in self.ids:
            if not self.active_ids[child_id] or child_id == self.parents[child_id]:
                continue
            parent_id = self.parents[child_id]
            children[parent_id.item()].append(child_id.item())

        max_child = max(len(children[node_id.item()]) for node_id in self.ids)
        self.children = -torch.ones((len(self.ids), max_child))
        self.num_children = torch.zeros(len(self.ids))

        for parent_id, child_ids in children.items():
            self.children[parent_id, : len(child_ids)] = torch.tensor(child_ids)
            self.num_children[parent_id] = len(child_ids)

    def destroy(self) -> None:
        """Used to destroy the graph and free up GPU memory."""
        for key in list(self.__dict__):
            if key not in ["plant", "params"]:
                delattr(self, key)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
