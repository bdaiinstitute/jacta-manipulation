# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
import torch
from scipy.spatial.transform import Rotation as R
from torch import FloatTensor

from dexterity.planner.dynamics.mujoco_dynamics import MujocoPlant
from dexterity.planner.dynamics.state_helpers import StateArray
from dexterity.planner.planner.parameter_container import ParameterContainer

isaac_to_mujoco_ixs = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
mujoco_to_isaac_ixs = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]


def isaac_to_mujoco(isaac: FloatTensor) -> FloatTensor:
    """Isaac to MuJoCo

    Args:
        isaac (FloatTensor): Isaac tensor

    Returns:
        FloatTensor: Mujoco tensor
    """
    return isaac[:, isaac_to_mujoco_ixs]


def mujoco_to_isaac(mujoco: FloatTensor) -> FloatTensor:
    """MuJoCo to Isaac

    Args:
        mujoco (FloatTensor): MuJoCo tensor

    Returns:
        FloatTensor: Isaac tensor
    """
    return mujoco[:, mujoco_to_isaac_ixs]


class LocomotionPlant(MujocoPlant):
    """LocomotionPlant"""

    def __init__(self, params: ParameterContainer) -> None:
        super().__init__(params=params)
        self.load_policy()

    def initialize(self) -> None:
        """Initialize Locomotion Plant

        Args:
            params (ParameterContainer): Parameters for initialization
        """
        super().initialize()
        self.action_dimension = 10  # TODO
        self.sensor_dimension = self.model.nsensordata + 12

    def load_policy(self) -> None:
        """Loads policy from the loaded params"""
        self.policy = torch.jit.load(self.params.policy_filepath).to(self.params.device)
        self.policy_output_scale = 0.2
        self.default_joint_pos_mujoco = torch.tensor([0.1, 0.9, -1.5, -0.1, 0.9, -1.5, 0.1, 1.1, -1.5, -0.1, 1.1, -1.5])
        self.default_joint_pos = mujoco_to_isaac(self.default_joint_pos_mujoco.unsqueeze(0))

    def process_observations(
        self,
        obs: dict,
        velocity_command: FloatTensor,
    ) -> FloatTensor:
        """Process the observations

        Args:
            obs (dict): Observations
            velocity_command (FloatTensor): Velocity command

        Returns:
            FloatTensor: Processed observations
        """
        obs_tensor = torch.cat(
            [
                obs["base_lin_vel"],
                obs["base_ang_vel"],
                obs["projected_gravity"],
                velocity_command,
                obs["joint_pos"],
                obs["joint_vel"],
                self.last_action,
            ],
            dim=-1,
        )
        return obs_tensor.float()

    def run_policy(
        self,
        base_command: FloatTensor,
        states: FloatTensor,
    ) -> FloatTensor:
        """Runs the policy

        Args:
            base_command (FloatTensor): Base command
            states (FloatTensor): States

        Returns:
            FloatTensor: Leg joint targets
        """
        policy_obs = {}

        # Use StateArray for indexing
        base_quat = states[:, StateArray.base_quat].cpu().numpy()
        quat_xyzw = base_quat[:, [1, 2, 3, 0]]  # w, x, y, z -> x, y, z, w
        base_rot = R.from_quat(quat_xyzw)
        base_rot_inv = base_rot.inv()

        base_lin_vel = states[:, StateArray.base_lin_vel].cpu().numpy()
        base_lin_vel = base_rot_inv.apply(base_lin_vel)
        policy_obs["base_lin_vel"] = torch.tensor(base_lin_vel)

        base_ang_vel = states[:, StateArray.base_ang_vel]
        policy_obs["base_ang_vel"] = base_ang_vel

        gravity_vector = [0, 0, -1]
        gravity_vector = base_rot_inv.apply(gravity_vector)
        policy_obs["projected_gravity"] = torch.tensor(gravity_vector)

        current_joint_pos = states[:, StateArray.legs]
        current_joint_pos = mujoco_to_isaac(current_joint_pos)

        joint_pos_diff = current_joint_pos - self.default_joint_pos
        policy_obs["joint_pos"] = joint_pos_diff

        joint_vel = states[:, StateArray.legs_vel]
        policy_obs["joint_vel"] = mujoco_to_isaac(joint_vel)

        input_obs = self.process_observations(
            obs=policy_obs,
            velocity_command=base_command,
        )
        with torch.no_grad():
            raw_output = self.policy(input_obs)
        self.last_action = raw_output

        leg_joint_targets = self.process_output(raw_output)

        return leg_joint_targets

    def process_output(self, raw_output: FloatTensor) -> FloatTensor:
        """Process the raw output

        Args:
            raw_output (FloatTensor): Raw output

        Returns:
            FloatTensor: Processed output
        """
        scaled_output = raw_output * self.policy_output_scale
        shifted_output = self.default_joint_pos + scaled_output
        return isaac_to_mujoco(shifted_output)

    def get_sensor(self, states: FloatTensor) -> FloatTensor:
        """Get sensor data from the states

        Args:
            states (FloatTensor): Vector of states

        Returns:
            FloatTensor: Sensor data
        """
        sensor = super().get_sensor(states)
        n_states = states.shape[0]
        empty_last_action = torch.zeros((n_states, 12), device=self.params.device)
        return torch.cat([sensor, empty_last_action], dim=-1)

    def dynamics(
        self,
        states: FloatTensor,
        action_trajectories: FloatTensor,
        info: dict,
    ) -> FloatTensor:
        """Dynamics

        Args:
            states (FloatTensor): States
            action_trajectories (FloatTensor): Action trajectories
            info (dict): Info

        Returns:
            FloatTensor: States, sensor data and state histories
        """
        if action_trajectories.dim() == 2:
            states = states.unsqueeze(0)
            action_trajectories = action_trajectories.unsqueeze(0)

        _, num_steps, _ = action_trajectories.shape

        state_histories = []
        self.last_action = info["sensor"][:, -12:]  # last_action from the previous step
        for t in range(num_steps):
            base_command = action_trajectories[:, t, :3]
            arm_command = action_trajectories[:, t, 3:]
            leg_joint_targets = self.run_policy(base_command, states)
            plant_action = torch.cat([leg_joint_targets, arm_command], dim=-1).unsqueeze(1)
            states, sensor_data, state_history = super().dynamics(states, plant_action)
            state_histories.append(state_history)

        state_histories = torch.cat(state_histories, dim=1)

        # Locomotion-only special case: sensor data also contains the last action
        final_sensordata = torch.cat([sensor_data, self.last_action], dim=-1)

        return states, final_sensordata, state_histories
