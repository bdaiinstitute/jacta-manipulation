# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
"""Mujoco Dynamics Submodule."""
from __future__ import annotations

from typing import Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np
import numpy.typing as npt
import torch
from mujoco import (
    MjModel,
    mj_differentiatePos,
    mj_fwdPosition,
    mj_fwdVelocity,
    mj_integratePos,
    mj_sensorPos,
    mj_sensorVel,
    mj_step1,
    rollout,
)
from torch import FloatTensor, IntTensor, tensor

from dexterity.planner.planner.parameter_container import ParameterContainer


def get_joint_dimensions(joint_ids: npt.ArrayLike, state_address: npt.ArrayLike, state_length: int) -> IntTensor:
    """Get the state vector indices corresponding to the given joint ids.

    Args:
        joint_ids: mujoco joint id
        state_address: array of joint start addresses
        state_length: total length of the state vector

    Returns:
        A tensor (nq,) with the indices in ``state_address`` corresponding to ``joint_ids``.
    """
    dims = []
    for idx in joint_ids:
        start = state_address[idx]  # start index in state
        if idx + 1 < len(state_address):
            end = state_address[idx + 1]  # end index in state
        else:
            end = state_length
        dims.extend(list(range(start, end)))
    return tensor(dims)


def decompose_state_dimensions(
    model: MjModel,
) -> Tuple[IntTensor, IntTensor, IntTensor, IntTensor]:
    """Decompose the states indices.

    Args:
        model: the model to extract states from

    Returns:
            Tuple: A Tuple containing:
                - actuated position indices
                - actuated velocity indices
                - unactuated position indices
                - unactuated velocity indices
    """
    nq = model.nq
    nv = model.nv

    joints = model.actuator_trnid[:, 0]
    pos_address = model.jnt_qposadr
    vel_address = model.jnt_dofadr

    actuated_pos = get_joint_dimensions(joints, pos_address, nq)
    actuated_vel = nq + get_joint_dimensions(joints, vel_address, nv)

    unactuated_pos = tensor([i for i in range(nq) if i not in actuated_pos])
    unactuated_vel = tensor([i for i in range(nq, nq + nv) if i not in actuated_vel])

    return actuated_pos, actuated_vel, unactuated_pos, unactuated_vel


def scale_distances(delta_states: FloatTensor, scaling: FloatTensor) -> FloatTensor:
    """Apply state distance cost matrix given by `scaling` to `delta_states`."""
    return torch.norm(delta_states @ scaling, dim=-1)


class MujocoPlant:
    """MujocoPlant object."""

    def __init__(self, *, params: Optional[ParameterContainer] = None, xml_model_path: Optional[str] = None) -> None:
        """Constructs a MujocoPlant object."""
        if params is None:
            self.params = ParameterContainer()
            self.params.model_filename = xml_model_path
            self.params.finite_diff_eps = 1e-3
        else:
            self.params = params
        self.initialize()

    def reset(self) -> None:
        """Resets a MujocoPlant."""
        self.initialize()

    def initialize(self) -> None:
        """Initializes MujocoPlant attributes.

        Args:
            params: the ``ParameterContainer`` with initialization params.
        """
        model_path = self.params.model_filename
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        self.model = model
        self.data = data
        if "action_time_step" not in self.params:
            self.params.action_time_step = pow(self.get_mass(), 1 / 6) / 10
        self.sim_time_step = model.opt.timestep
        if "num_substeps" not in self.params:
            self.params.num_substeps = int(np.round(self.params.action_time_step / self.sim_time_step))
        self.state_dimension = model.nq + model.nv + model.na
        self.state_derivative_dimension = 2 * model.nv + model.na
        self.action_dimension = model.nu
        self.sensor_dimension = model.nsensordata

        self.actuated_pos, self.actuated_vel, self.unactuated_pos, self.unactuated_vel = decompose_state_dimensions(
            model
        )
        self.unactuated_pos_difference = self.unactuated_vel - model.nq
        self.actuated_state = torch.cat((self.actuated_pos, self.actuated_vel))
        self.unactuated_state = torch.cat((self.unactuated_pos, self.unactuated_vel))
        self.get_quat_indices()

    def dynamics(
        self,
        states: FloatTensor,
        action_trajectories: FloatTensor,
        info: dict | None = None,
    ) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:
        """Conditions on the size of the states/actions and calls the appropriate singular or parallel dynamics.

        Args:
            states: (nx,) or (num_envs, nx) sized vector of states
            action_trajectories: (num_steps, na) or (num_envs, num_steps, na) array containing
            the start and end action vectors of the desired trajectory.
            action_time_step: the hold time for the action.
            info: additional information to pass to the dynamics function.

        Returns:
            A tuple of (next state, sensor data, intermediate states)
        """
        states_dim, actions_dim = states.ndim, action_trajectories.ndim
        if (states_dim, actions_dim) == (1, 2):
            num_envs = 0
            states = torch.hstack((torch.ones((1, 1)) * self.data.time, states.unsqueeze(0)))
            action_trajectories = action_trajectories.unsqueeze(0)
        elif (states_dim, actions_dim) == (2, 3):
            num_envs = states.shape[0]
            states = torch.hstack((torch.ones((num_envs, 1)) * self.data.time, states))
        else:
            raise ValueError("Invalid dimensions for states and action_trajectories.")

        return self._dynamics(states, action_trajectories, num_envs=num_envs)

    def _get_gradient_placeholders(
        self,
        size: Optional[int] = None,
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        if size is None:
            state_gradient_state = np.eye(self.state_derivative_dimension)
            state_gradient_control = np.zeros((self.state_derivative_dimension, self.action_dimension))
            sensor_gradient_state = np.zeros((self.sensor_dimension, self.state_derivative_dimension))
            sensor_gradient_control = np.zeros((self.sensor_dimension, self.action_dimension))
        else:
            state_gradient_state = np.eye(self.state_derivative_dimension)
            state_gradient_state = np.tile(state_gradient_state, (size, 1, 1))
            state_gradient_control = np.zeros((size, self.state_derivative_dimension, self.action_dimension))
            sensor_gradient_state = np.zeros((size, self.sensor_dimension, self.state_derivative_dimension))
            sensor_gradient_control = np.zeros((size, self.sensor_dimension, self.action_dimension))

        return state_gradient_state, state_gradient_control, sensor_gradient_state, sensor_gradient_control

    def get_gradients(
        self,
        states: FloatTensor,
        actions: FloatTensor,
    ) -> Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor]:
        """Computes the dynamics gradients.

        Args:
            states: (nx,) or (num_envs, nx) sized vector of states
            actions: (na,) or (num_envs, na) array containing the action vectors.

        Returns:
            Tuple: A Tuple containing:
                - state_gradients_state: (nx, nx) or (num_envs, nx, nx)
                - state_gradients_control: (nx, nu) or (num_envs, nx, nu)
                - sensor_gradients_state: (ns, nx) or (num_envs, ns, nx)
                - sensor_gradients_control: (ns, nu) or (num_envs, ns, nu)
        """
        states_dim, actions_dim = states.ndim, actions.ndim
        match (states_dim, actions_dim):
            case (1, 1):
                (
                    state_gradients_state,
                    state_gradients_control,
                    sensor_gradients_state,
                    sensor_gradients_control,
                ) = self._get_gradient_placeholders()
                self.set_state(states)
                self.set_action(actions)
                mujoco.mjd_transitionFD(
                    self.model,
                    self.data,
                    self.params.finite_diff_eps,
                    True,
                    state_gradients_state,
                    state_gradients_control,
                    sensor_gradients_state,
                    sensor_gradients_control,
                )
            case (2, 2):
                num_envs = states.shape[0]
                (
                    state_gradients_state,
                    state_gradients_control,
                    sensor_gradients_state,
                    sensor_gradients_control,
                ) = self._get_gradient_placeholders(size=num_envs)
                for i in range(num_envs):
                    self.set_state(states[i, :])
                    self.set_action(actions[i, :])
                    mujoco.mjd_transitionFD(
                        self.model,
                        self.data,
                        self.params.finite_diff_eps,
                        True,
                        state_gradients_state[i, :, :],
                        state_gradients_control[i, :, :],
                        sensor_gradients_state[i, :, :],
                        sensor_gradients_control[i, :, :],
                    )
            case _:
                raise ValueError("Bad dimensions")

        # TODO slecleach maybe not necessary to specify dtype.
        state_gradients_state = tensor(state_gradients_state, dtype=torch.float32)
        state_gradients_control = tensor(state_gradients_control, dtype=torch.float32)
        sensor_gradients_state = tensor(sensor_gradients_state, dtype=torch.float32)
        sensor_gradients_control = tensor(sensor_gradients_control, dtype=torch.float32)
        return state_gradients_state, state_gradients_control, sensor_gradients_state, sensor_gradients_control

    def set_state(self, state: FloatTensor) -> None:
        """Set plant state."""
        self.data.qpos = state[0 : self.model.nq].cpu().numpy()
        self.data.qvel = state[self.model.nq :].cpu().numpy()

    def get_state(self) -> FloatTensor:
        """Get plant state."""
        qpos = tensor(self.data.qpos, dtype=torch.float32)
        qvel = tensor(self.data.qvel, dtype=torch.float32)
        state = torch.cat([qpos, qvel])
        return state

    def set_action(self, action: FloatTensor) -> None:
        """Set plant action."""
        self.data.ctrl = action.cpu().numpy()

    def get_action(self) -> FloatTensor:
        """Get last plant action."""
        return tensor(self.data.ctrl, dtype=torch.float32)

    def update_sensor(self) -> None:
        """Update plant sensor measurement.."""
        mj_fwdPosition(self.model, self.data)
        mj_sensorPos(self.model, self.data)
        mj_fwdVelocity(self.model, self.data)
        mj_sensorVel(self.model, self.data)

    def get_sensor(self, states: FloatTensor) -> FloatTensor:
        # TODO: the states input is sometimes a torch tensor not an FloatTensor
        """Update plant sensor measurement.

        This only supports position- and velocity-based sensors, NOT ACCLERATION-BASED sensors.
        We use the minimal set of computations extracted from mj_step1, see the link below for more details:
        https://mujoco.readthedocs.io/en/latest/programming/simulation.html?highlight=mj_step1#simulation-loop
        # TODO: slecleach add support for acceleration-based sensors
        Finally, returns the sensor measurement.

        Args:
            states: (nx,) or (num_envs, nx) sized vector of states

        Returns:
            Sensor data (nsensordata,) or (num_envs, nsensordata)
        """
        states_dim = states.ndim
        if states_dim == 1:
            self.set_state(states)
            self.update_sensor()
            return tensor(self.data.sensordata, dtype=torch.float32)
        elif states_dim == 2:
            num_envs = states.shape[0]
            sensor = torch.zeros((num_envs, self.model.nsensordata))
            for i in range(num_envs):
                self.set_state(states[i, :])
                self.update_sensor()
                sensor[i, :] = tensor(self.data.sensordata, dtype=torch.float32)
            return sensor
        else:
            raise ValueError("Input states must have dimensionality 1 or 2.")

    def scaled_distances_to(self, states: FloatTensor, target_states: FloatTensor) -> FloatTensor:
        """Get scaled distance between `states` and `target_states`."""
        delta_states = self.state_difference(states, target_states)
        return scale_distances(delta_states, self.params.reward_distance_scaling_sqrt)

    def state_difference(self, s1: FloatTensor, s2: FloatTensor, h: float = 1.0) -> FloatTensor:
        """Compute finite-difference velocity given two state vectors and a time step h.

        ds = (s2 - s1) / h

        Args:
            s1: first state vector.
            s2: second state vector.
            h: time step.

        Returns:
            ds, or the finite difference velocity.
        """
        model = self.model
        nq = model.nq
        nv = model.nv
        s1 = s1.cpu().numpy()
        s2 = s2.cpu().numpy()

        s1_dim = s1.ndim
        s2_dim = s2.ndim
        if nq == nv and s1_dim == s2_dim:
            ds = (s2 - s1) / h
        else:
            match (s1_dim, s2_dim):
                case (1, 1):  # 1 to 1
                    ds = np.zeros(self.state_derivative_dimension)
                    mj_differentiatePos(model, ds[:nv], h, s1[:nq], s2[:nq])
                    ds[nv:] = (s2[nq:] - s1[nq:]) / h
                case (2, 1):  # n to 1, TODO vectorize
                    num_states = s1.shape[0]
                    ds = np.zeros((num_states, self.state_derivative_dimension))
                    for i in range(num_states):
                        mj_differentiatePos(model, ds[i, :nv], h, s1[i, :nq], s2[:nq])
                        ds[i, nv:] = (s2[nq:] - s1[i, nq:]) / h
                case (1, 2):  # 1 to n, TODO vectorize
                    num_states = s2.shape[0]
                    ds = np.zeros((num_states, self.state_derivative_dimension))
                    for i in range(num_states):
                        mj_differentiatePos(model, ds[i, :nv], h, s1[:nq], s2[i, :nq])
                        ds[i, nv:] = (s2[i, nq:] - s1[nq:]) / h
                case (2, 2):  # n to n, TODO vectorize
                    num_states = s1.shape[0]
                    ds = np.zeros((num_states, self.state_derivative_dimension))
                    for i in range(num_states):
                        mj_differentiatePos(model, ds[i, :nv], h, s1[i, :nq], s2[i, :nq])
                        ds[i, nv:] = (s2[i, nq:] - s1[i, nq:]) / h
                case (3, 2):
                    num_states_1, num_states_2 = s1.shape[0], s1.shape[1]
                    ds = np.zeros((num_states_1, num_states_2, self.state_derivative_dimension))
                    for i in range(num_states_1):
                        for j in range(num_states_2):
                            mj_differentiatePos(model, ds[i, j, :nv], h, s1[i, j, :nq], s2[i, :nq])
                            ds[i, j, nv:] = (s2[i, nq:] - s1[i, j, nq:]) / h
                case _:
                    raise ValueError("Invalid dimensions.")

        return tensor(ds, dtype=torch.float32)

    def state_addition(self, s1: FloatTensor, ds: FloatTensor, h: float = 1.0) -> FloatTensor:
        """Integrate forward a state s with a velocity ds for a time step h.

        s2 = s1 + h * ds

        Args:
            s1: state vector to integrate forward.
            ds: velocity.
            h: time step.

        Returns:
            s2, or the resulting integrated state.
        """
        model = self.model
        nq = model.nq
        nv = model.nv
        na = model.na
        s1 = s1.cpu().numpy()
        ds = ds.cpu().numpy()

        if nq == nv:
            s2 = s1 + h * ds
        else:
            s2 = np.zeros(nq + nv + na)
            s2[:] = s1[:]
            mj_integratePos(model, s2[:nq], ds[:nv], h)
            s2[nq:] = s1[nq:] + h * ds[nv:]
        return tensor(s2, dtype=torch.float32)

    def _dynamics(
        self, state: FloatTensor, action_trajectory: FloatTensor, num_envs: int
    ) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:
        """Implementation of the dynamics function in Mujoco.

        Args:
            state: initial states.
            action_trajectory: an action trajectory of size (num_steps, na).
            num_envs: number of envs

        Returns:
            final_states: (nx,) or (num_envs, nx)
            final_sensordata: (nsensordata,) or (num_envs, nsensordata)
            state_trajectories: (num_steps, nx) or (num_envs, num_steps, nx)
        """
        state_trajectories, sensordata = rollout.rollout(
            self.model, self.data, state.cpu().numpy(), action_trajectory.cpu().numpy()
        )
        state_trajectories = tensor(state_trajectories[:, :, 1:], dtype=torch.float32)
        final_states = state_trajectories[:, -1]
        sensordata = tensor(sensordata, dtype=torch.float32)
        final_sensordata = sensordata[:, -1]
        if num_envs == 0:
            final_states = final_states.squeeze()
            final_sensordata = final_sensordata.squeeze()
            state_trajectories = state_trajectories.squeeze()
        return final_states, final_sensordata, state_trajectories

    def get_mass(self) -> float:
        """Retrieve the total body mass."""
        mass = 0
        for idx in range(self.model.nbody):
            mass += self.model.body_mass[idx]

        return mass

    def get_quat_indices(self) -> None:
        """Stores the indices of the state corresponding to quaternion in class field."""
        quat_indices = []

        jnt_types = self.model.jnt_type
        jnt_start_indices = self.model.jnt_qposadr
        for i, jnt in enumerate(jnt_types):
            if jnt == 0:  # mjJNT_FREE expressed in global pos and orientation
                jnt_start_idx = jnt_start_indices[i]
                quat_start_idx = jnt_start_idx + 3
                quat_end_idx = quat_start_idx + 4
                quat_idx = torch.tensor([i for i in range(quat_start_idx, quat_end_idx)])
                quat_indices.append(quat_idx)
        self.quat_indices = quat_indices

    def normalize_state(self, states: torch.FloatTensor) -> torch.FloatTensor:
        """Normalize quaternion part of state.

        Args:
            states (torch.FloatTensor): (num_envs, nx) tensor of randomly sampled states
        Outputs:
            torch.FloatTensor: (num_envs, nx) tensor with quaternion portions normalized
        """
        if self.quat_indices:
            normalized_states = states.clone()

            # Extract quaternion portion of state
            quat_indices = torch.cat(self.quat_indices, dim=0)
            quats = states[:, quat_indices].reshape(-1, 4)

            # Compute norms along the last dimension
            quat_norms = torch.norm(quats, p=2, dim=-1)

            # Set norm to 1 where it is zero
            non_zero_norm = quat_norms != 0
            quat_norms[~non_zero_norm] = 1

            # Normalize quaternion tensor
            expanded_norms = quat_norms.unsqueeze(1).expand_as(quats)
            normalized_quats = quats / expanded_norms

            # Flatten and replace existing quat with normalized quat
            flattened_normalized_quats = normalized_quats.reshape(-1, len(self.quat_indices) * 4)
            normalized_states[:, quat_indices] = flattened_normalized_quats

            return normalized_states
        return states

    def get_collision_free(self, states: FloatTensor) -> Optional[FloatTensor]:
        """Get collision free states.

        Args:
            states (FloatTensor): (num_envs, nx) tensor of randomly sampled states
        Outputs:
            Optional[FloatTensor]: tensor of all collision free states (or None if none exist)
        """
        collision_free_states = tuple()  # type: tuple
        collision_threshold = 1e-3
        for _, state in enumerate(states):
            self.set_state(state)
            mj_step1(self.model, self.data)
            sdf = self.data.contact.dist

            # Check for penetration rather than contact
            if not np.any(sdf < collision_threshold):
                collision_free_states = (*collision_free_states, state)
        if collision_free_states:
            return torch.stack(collision_free_states)  # type:ignore[unreachable]
        return None
