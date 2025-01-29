# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import numpy as np
import torch

from dexterity.planner.dynamics.locomotion_plant import LocomotionPlant
from dexterity.planner.dynamics.mujoco_dynamics import MujocoPlant
from dexterity.planner.planner.action_processor import SpotFloatingActionProcessor, SpotWholebodyActionProcessor
from dexterity.planner.planner.parameter_container import ParameterContainer


def setup_spot_floating() -> tuple[MujocoPlant, SpotFloatingActionProcessor]:
    params = ParameterContainer("dexterity/examples/planner/config/spot_floating_tire.yml")

    object_state = [5.0, 5.0, 0.3, 1, 0, 0, 0]
    robot_state = [0.0, 0.0, 0.0, 0.0, -0.9, 1.8, 0.0, 0.6, 0.0, -1.54]
    velocities = [0.0] * 16

    params.update(
        {
            "start_state": object_state + robot_state + velocities,
            "goal_state": object_state + robot_state + velocities,
            "action_bound_lower": [-10, -10, -100, -2.62, -np.pi, 0, -2.79, -1.83, -2.87, -1.57],
            "action_bound_upper": [+10, +10, +100, +2.62, +np.pi, 0, +2.79, +1.83, +2.87, +1.57],
            "action_range": [0.3] * 3 + [0.0] * 7,
            "action_start_mode": "ActionMode.RELATIVE_TO_CURRENT_STATE",
        }
    )

    plant = MujocoPlant(params=params)
    action_processor = SpotFloatingActionProcessor(params, plant.actuated_pos)
    plant.set_state(params.start_state)

    return plant, action_processor


def turn_towards(point: np.ndarray, plant: MujocoPlant) -> torch.FloatTensor:
    """Produce turning action until the robot faces the point."""
    current_state = plant.get_state().cpu().numpy()
    robot_x, robot_y, robot_yaw = current_state[7:10]
    angle = np.arctan2(point[1] - robot_y, point[0] - robot_x)
    delta_base = [0.0, 0.0, angle - robot_yaw]
    delta_arm = [0.0] * 7
    return torch.tensor([delta_base + delta_arm])


def move_forward(point: np.ndarray, plant: MujocoPlant) -> torch.FloatTensor:
    """Produce forward action until the robot reaches the point."""
    current_state = plant.get_state().cpu().numpy()
    robot_x, robot_y, robot_theta = current_state[7:10]

    vector = point - np.array([robot_x, robot_y])
    distance = np.linalg.norm(vector)
    reference_direction = np.array([np.cos(robot_theta), np.sin(robot_theta)])
    signed_distance = distance * np.sign(np.dot(vector, reference_direction))

    delta_base = [signed_distance, 0.0, 0.0]
    delta_arm = [0.0] * 7
    return torch.tensor([delta_base + delta_arm])


def test_spot_floating_egomotion() -> None:
    """Use SpotFloating robot to move to the point in 2 stages: turn and move forward."""
    plant, action_processor = setup_spot_floating()
    current_state = plant.get_state()

    goal_point = np.array([2.0, 2.0])
    turn_duration = 20
    total_duration = 80

    for t in range(total_duration):
        relative_actions = turn_towards(goal_point, plant) if t < turn_duration else move_forward(goal_point, plant)
        relative_actions = torch.clamp(relative_actions, -plant.params.action_range, plant.params.action_range)

        _, _, action_trajectory = action_processor(
            relative_actions=relative_actions,
            current_states=plant.get_state().unsqueeze(0),
        )

        current_state, *_ = plant.dynamics(current_state, action_trajectory.squeeze(0))

    robot_xy = current_state.cpu().numpy()[7:9]
    distance_to_goal = np.linalg.norm(goal_point - robot_xy)

    assert distance_to_goal < 0.1, f"Distance to goal: {distance_to_goal}"


def setup_spot_locomotion() -> tuple[MujocoPlant, SpotFloatingActionProcessor]:
    params = ParameterContainer("dexterity/examples/planner/config/spot_tire.yml")

    object_state = [5.0, 5.0, 0.3, 1, 0, 0, 0]
    robot_base = [0.0, 0.0, 0.51, 1, 0, 0, 0]
    robot_legs = [0.1, 0.9, -1.5, -0.1, 0.9, -1.5, 0.1, 1.1, -1.5, -0.1, 1.1, -1.5]
    robot_arm = [0.0, -0.9, 1.8, 0.0, 0.6, 0.0, -1.54]
    velocities = [0.0] * 31

    params.update(
        {
            "start_state": robot_base + robot_legs + robot_arm + object_state + velocities,
            "goal_state": robot_base + robot_legs + robot_arm + object_state + velocities,
            "action_start_mode": "ActionMode.RELATIVE_TO_CURRENT_STATE",
            "action_time_step": 1.0 / 50.0,
            "policy_filename": "locomotion.pt",
        }
    )

    plant = LocomotionPlant(params)
    action_processor = SpotWholebodyActionProcessor(params, plant.actuated_pos)
    plant.set_state(params.start_state)

    return plant, action_processor


def test_spot_locomotion_capabilities(meters: float = 3.0) -> None:
    """Use SpotWholebody robot to move forward until it reaches the specified distance."""
    plant, action_processor = setup_spot_locomotion()

    states = plant.get_state().unsqueeze(0)

    info = {
        "sensor": plant.get_sensor(states),
    }

    for _ in range(500):
        fwd = 1.0
        delta_base = [fwd, 0.0, 0.0]
        delta_arm = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        relative_actions = torch.tensor([delta_base + delta_arm])

        _, _, action_trajectories = action_processor(
            relative_actions=relative_actions,
            current_states=states,
        )

        states, sensor, _ = plant.dynamics(
            states,
            action_trajectories,
            info,
        )
        info["sensor"] = sensor

        robot_x = states.cpu().numpy()[0, 0]  # env = 0; robot_base_x = 0

        if robot_x > meters:
            break

    assert robot_x > meters, f"Robot only reached: {robot_x}"
