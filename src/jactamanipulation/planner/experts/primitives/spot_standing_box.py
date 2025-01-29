# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
"""Spot Standing Box specific action primitives (e.g. grasping)."""

from typing import Dict

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

from jactamanipulation.planner.dynamics.mujoco_dynamics import MujocoPlant
from jactamanipulation.planner.planner.linear_algebra import transformation_matrix


def spot_ik_mujoco(plant: MujocoPlant, dx: np.ndarray, ee_body: str = "arm_link_fngr") -> np.ndarray:
    """Retrieve IK for a given end effector."""
    gripper = plant.data.body(ee_body)
    jacp, jacr = np.zeros((3, plant.model.nv)), np.zeros((3, plant.model.nv))
    mujoco.mj_jacBody(plant.model, plant.data, jacp, jacr, gripper.id)
    Jee = np.vstack([jacp, jacr])
    pinvJee = np.linalg.pinv(Jee)
    dq = pinvJee @ dx
    return dq


def compute_grasp_pose_for_handle_top(plant: MujocoPlant) -> np.ndarray:
    """Returns the change in joint configuration in order to grasp the box handle.

    The grasp pose is hard-coded based on the specific spot_standing_box domain.
    """
    handle = plant.data.geom("handle_top")
    gripper = plant.data.body("arm_link_fngr")

    world_t_gripper = transformation_matrix(gripper.xmat.reshape(3, 3), gripper.xpos)
    world_t_handle = transformation_matrix(handle.xmat.reshape(3, 3), handle.xpos)

    handle_t_gripper_goal_rot = R.from_euler("xyz", [-np.pi / 2, 0, 0]).as_matrix()
    handle_t_gripper_goal = transformation_matrix(handle_t_gripper_goal_rot, [0.05, 0, 0])
    world_t_gripper_goal = world_t_handle @ handle_t_gripper_goal

    # Compute change in gripper pose needed to reach gripper goal
    dpos = world_t_gripper_goal[:3, 3] - world_t_gripper[:3, 3]
    world_rot_gripper = world_t_gripper[:3, :3]
    world_rot_gripper_goal = world_t_gripper_goal[:3, :3]

    gripper_rot_gripper_goal = np.linalg.inv(world_rot_gripper) @ world_rot_gripper_goal
    drot = R.from_matrix(gripper_rot_gripper_goal).as_euler("xyz")
    dx = np.concatenate([dpos, drot])

    # Compute corresponding joint config change by IK
    dq = spot_ik_mujoco(plant, dx, ee_body="arm_link_fngr")
    return dq[6:]


def compute_grasp_actions(plant: MujocoPlant, **kwargs: Dict) -> np.ndarray:
    """Returns a trajectory to move the arm to the box handle.

    The trajectory terminates when the gripper has closed. The trajectory will
    have three parts: (1) open gripper (2) move arm (3) close gripper.  Each part
    is represented by a tuple: (start_action, end_action, action_time).
    """
    open_gripper_time = kwargs.get("open_gripper_time", 0.6)
    close_gripper_time = kwargs.get("close_gripper_time", 0.6)
    move_arm_time = kwargs.get("move_arm_time", 1.0)

    # gripper position: 0 (close), -1.54 (open); values greater than 0 are invalid
    open_gripper_action = (plant.data.qpos[7:], np.concatenate([plant.data.qpos[7:-1], [-1.54]]), open_gripper_time)

    dq_move_arm = compute_grasp_pose_for_handle_top(plant)
    dq_move_arm[-1] = 0.0  # don't move gripper
    move_arm_action = (open_gripper_action[1], open_gripper_action[1] + dq_move_arm, move_arm_time)
    close_gripper_action = (
        move_arm_action[1],
        np.concatenate([move_arm_action[1][:-1], [1.54]]),
        close_gripper_time,
    )

    return [open_gripper_action, move_arm_action, close_gripper_action]
