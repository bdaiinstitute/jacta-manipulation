# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import trange

from dexterity.planner.dynamics.mujoco_dynamics import MujocoPlant
from dexterity.planner.planner.parameter_container import ParameterContainer
from dexterity.planner.visuals.mujoco_visualizer import MujocoRenderer


@dataclass
class TireGeneratorConfig:
    """Configuration for state cache generation."""

    task: Literal["spot_tire", "spot_floating_tire"] = "spot_tire"
    cache_size: int = 10_000
    enable_rendering: bool = False
    radius_range: tuple[float, float] = (-np.pi / 3, np.pi / 3)
    distance_range: tuple[float, float] = (0.8, 1.2)
    object_z: float = 0.3
    linear_velocity_range: tuple[float, float] = (1.0, 1.0)
    angular_velocity_range: tuple[float, float] = (-2.0, 2.0)
    randomize_velocity: bool = True


#  TODO move spot arm ik to kinematics once needed elsewhere
@dataclass
class SpotArmIK:
    """Inverse kinematics for the Spot arm."""

    plant: MujocoPlant
    params: ParameterContainer
    actuated_pos: torch.Tensor
    max_iterations: int = 200
    integration_dt: float = 10.0
    damping: float = 1e-4
    enable_orientation_control: bool = True
    position_tolerance: float = 1e-2
    orientation_tolerance: float = 1e-2
    progress_threshold: float = 1e-4
    max_no_progress_steps: int = 10

    def __post_init__(self) -> None:
        self.end_effector_site_id = self.plant.model.site("site_arm_link_wr1").id
        self.actuated_joint_ids = self.actuated_pos[-7:].cpu().numpy()
        all_actuated_joints = list(range(self.plant.data.ctrl.shape[0]))
        self.actuator_ids = all_actuated_joints[-7:]

    def solve(self, target_xpos: np.ndarray, target_xquat: np.ndarray) -> list[float]:
        """Solve the inverse kinematics problem for the given target position and orientation.

        Args:
            target_xpos (np.ndarray): Target position.
            target_xquat (np.ndarray): Target orientation as a quaternion.

        Returns:
            list[float]: Solved joint angles.
        """
        data = self.plant.data
        model = self.plant.model
        end_effector_site_id = self.end_effector_site_id

        jac = np.zeros((6, model.nv))
        diag = self.damping * np.eye(6)
        error = np.zeros(6)
        error_pos = error[:3]
        error_ori = error[3:]
        site_quat = np.zeros(4)
        site_quat_conj = np.zeros(4)
        error_quat = np.zeros(4)

        no_progress_steps = 0
        prev_error_pos_norm = np.inf

        for _ in range(self.max_iterations):
            error_pos[:] = target_xpos - data.site(end_effector_site_id).xpos

            mujoco.mju_mat2Quat(site_quat, data.site(end_effector_site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)

            if self.enable_orientation_control:
                mujoco.mju_mulQuat(error_quat, target_xquat, site_quat_conj)
                mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)

            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], end_effector_site_id)
            dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)

            q = data.qpos.copy()
            mujoco.mj_integratePos(model, q, dq, self.integration_dt)

            arange = self.params.action_range[-7:].cpu().numpy()
            arm_control = np.clip(q[self.actuated_joint_ids], -arange, +arange)
            data.ctrl[self.actuator_ids] = arm_control

            mujoco.mj_step(model, data)

            error_pos_norm = np.linalg.norm(error_pos)
            if error_pos_norm < self.position_tolerance and np.linalg.norm(error_ori) < self.orientation_tolerance:
                break

            if abs(prev_error_pos_norm - error_pos_norm) < self.progress_threshold:
                no_progress_steps += 1
                if no_progress_steps >= self.max_no_progress_steps:
                    break
            else:
                no_progress_steps = 0

            prev_error_pos_norm = error_pos_norm

        return list(data.qpos[self.actuated_joint_ids])


@dataclass
class TireStateCacheGenerator:
    """Generator for tire state cache."""

    task_name: str
    config: TireGeneratorConfig

    def __post_init__(self) -> None:
        self.params = ParameterContainer(f"dexterity/examples/planner/config/{self.task_name}.yml")
        self.plant = MujocoPlant(params=self.params)
        self.arm_ik = SpotArmIK(self.plant, self.params, self.plant.actuated_pos)
        self.params.state_cache_folder.mkdir(parents=True, exist_ok=True)
        self.renderer = MujocoRenderer(plant=self.plant) if self.config.enable_rendering else None

    def generate_cache(self) -> torch.Tensor:
        """Generate a cache of states for the tire.

        Returns:
            torch.Tensor: Generated state cache.
        """
        state_cache = torch.zeros((self.config.cache_size, self.plant.model.nq + self.plant.model.nv))

        for state_i in trange(self.config.cache_size):
            state = self._generate_single_state()
            if self.config.enable_rendering:
                self._render_state(state)
            state_cache[state_i] = state

        return state_cache

    def set_floating_spot_cache(self, cache: torch.Tensor) -> None:
        """Set the floating spot cache.

        Args:
            cache (torch.Tensor): The cache to set.
        """
        self.floating_spot_cache = cache

    def _generate_single_state(self) -> torch.Tensor:
        base_pose = self.default_base_pose
        object_pose = self._randomize_object_pose()
        state = torch.cat([object_pose, base_pose, self.default_arm_pose, self.default_velocities])
        arm_pose = self._generate_arm_pose(state)
        state = torch.cat([object_pose, base_pose, arm_pose, self.default_velocities])
        state = self._apply_velocity_perturbation(state) if self.config.randomize_velocity else state
        return state

    def _randomize_object_pose(self) -> torch.Tensor:
        radius = np.random.uniform(*self.config.radius_range)
        distance = np.random.uniform(*self.config.distance_range)
        xyz = torch.tensor([distance * np.cos(radius), distance * np.sin(radius), self.config.object_z])

        pitch = np.random.uniform(-np.pi, np.pi)
        yaw = np.random.uniform(-np.pi, np.pi)
        rot = R.from_euler("z", yaw) * R.from_euler("y", pitch)
        quat = torch.tensor([rot.as_quat()[3], rot.as_quat()[0], rot.as_quat()[1], rot.as_quat()[2]])
        return torch.cat([xyz, quat])

    def _generate_arm_pose(self, state: torch.Tensor) -> torch.Tensor:
        object_xyz = state[:3]
        if "floating" in self.task_name:
            self.plant.set_state(state)
            target_xpos = object_xyz.cpu().numpy() + np.array([0.0, 0.0, 0.4])
            target_xquat = np.array([1.0, 0.0, 0.0, 0.0])
            return torch.tensor(self.arm_ik.solve(target_xpos, target_xquat))
        else:
            return self._closest_arm_pose(object_xyz)

    def _closest_arm_pose(self, object_xyz: torch.Tensor) -> torch.Tensor:
        distances = torch.norm(self.floating_spot_cache[:, :3] - object_xyz, dim=-1)
        closest_state = self.floating_spot_cache[distances.argmin()]
        return closest_state[10:17]  # Adjust indices as needed

    @property
    def default_arm_pose(self) -> torch.Tensor:
        """Get the default arm pose.

        Returns:
            torch.Tensor: Default arm pose.
        """
        return torch.tensor([0.0, -0.9, 1.8, 0.0, -0.9, 0.0, 0.0])

    @property
    def default_base_pose(self) -> torch.Tensor:
        """Get the default base pose.

        Returns:
            torch.Tensor: Default base pose.
        """
        if "floating" in self.task_name:
            return torch.zeros(3)
        else:
            base_xyz = torch.tensor([0.0, 0.0, 0.51])
            base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
            legs = torch.tensor([+0.1, 0.9, -1.5, -0.1, 0.9, -1.5, +0.1, 1.1, -1.5, -0.1, 1.1, -1.5])
            return torch.cat([base_xyz, base_quat, legs])

    @property
    def default_velocities(self) -> torch.Tensor:
        """Get the default velocities.

        Returns:
            torch.Tensor: Default velocities.
        """
        return torch.zeros(self.plant.model.nv)

    def _apply_velocity_perturbation(self, state: torch.Tensor) -> torch.Tensor:
        linear_vel_slice = slice(self.plant.model.nq, self.plant.model.nq + 3)
        angular_vel_slice = slice(self.plant.model.nq + 3, self.plant.model.nq + 6)

        object_orn = state[3:7].cpu().numpy()
        tire_rot = R.from_quat(object_orn[[1, 2, 3, 0]])

        # Ignore tire's pitch rotation
        euler = tire_rot.as_euler("xyz", degrees=False)
        euler[1] = 0
        adjusted_tire_rot = R.from_euler("xyz", euler, degrees=False)

        linear_velocity = torch.tensor(
            adjusted_tire_rot.apply([np.random.uniform(*self.config.linear_velocity_range), 0.0, 0.0])
        )
        angular_velocity = torch.tensor(
            tire_rot.inv().apply([0.0, 0.0, np.random.uniform(*self.config.angular_velocity_range)])
        )

        state[linear_vel_slice] = linear_velocity
        state[angular_vel_slice] = angular_velocity

        return state

    def _render_state(self, state: torch.Tensor) -> None:
        self.renderer.reset()
        self.renderer.render(state[: self.plant.model.nq].unsqueeze(0).cpu().numpy())

        if self.config.randomize_velocity:
            for _ in range(20):
                state, _, state_traj = self.plant.dynamics(state, torch.zeros((10, self.plant.model.nu)))
                for s in state_traj:
                    self.renderer.render(s[: self.plant.model.nq].cpu().numpy())

        self.renderer.play(wait_for_input=True)
