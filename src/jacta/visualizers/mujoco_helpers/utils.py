# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

"""Class for rendering Mujoco trajectories in meshcat"""

import time
from typing import List

import mujoco
import numpy as np
from mujoco import MjModel
from torch import FloatTensor

from jacta.visualizers.mujoco_helpers.dynamics import MujocoPlant


def get_sensor_name(model: MjModel, sensorid: int) -> str:
    """Return name of the sensor with given ID from MjModel."""
    index = model.name_sensoradr[sensorid]
    end = model.names.find(b"\x00", index)
    name = model.names[index:end].decode("utf-8")
    if len(name) == 0:
        name = f"sensor{sensorid}"
    return name


def get_mesh_data(model: MjModel, meshid: int) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve the vertices and faces of a specified mesh from a MuJoCo model.

    Args:
        model : MjModel The MuJoCo model containing the mesh data.
        meshid : int The index of the mesh to retrieve.

    Result:
        tuple[np.ndarray, np.ndarray]
        Vertices (N, 3) and faces (M, 3) of the mesh.
    """
    vertadr = model.mesh_vertadr[meshid]
    vertnum = model.mesh_vertnum[meshid]
    vertices = model.mesh_vert[vertadr : vertadr + vertnum, :]

    faceadr = model.mesh_faceadr[meshid]
    facenum = model.mesh_facenum[meshid]
    faces = model.mesh_face[faceadr : faceadr + facenum]
    return vertices, faces


def is_trace_sensor(model: MjModel, sensorid: int) -> bool:
    sensor_name = get_sensor_name(model, sensorid)
    return (
        model.sensor_type[sensorid] == 25
        and model.sensor_datatype[sensorid] == 0
        and model.sensor_dim[sensorid] == 3
        and "trace" in sensor_name
    )


def count_trace_sensors(model: MjModel) -> int:
    """Count the number of trace sensors of a given mujoco model."""
    num_traces = 0
    for id in range(model.nsensor):
        num_traces += is_trace_sensor(model, id)
    return num_traces


def get_trace_sensors(model: MjModel) -> List[int]:
    return [id for id in range(model.nsensor) if is_trace_sensor(model, id)]


def set_mocap_pose(data: mujoco.MjData, mocap_id: int, pose: np.ndarray) -> None:
    """Set the position of the mocap element."""
    if mocap_id < 0 or mocap_id >= data.mocap_pos.shape[0]:
        raise ValueError(f"Invalid mocap ID: {mocap_id}")

    # Set the position of the mocap element
    data.mocap_pos[mocap_id] = pose[0:3]
    data.mocap_quat[mocap_id] = pose[3:7]


def set_mocap_poses(
    data: mujoco.MjData, mocap_ids: list[int], poses: list[np.ndarray]
) -> None:
    """Set the positions of the mocap elements."""
    for mocap_id, pose in zip(mocap_ids, poses, strict=False):
        if pose is not None and mocap_id < data.mocap_pos.shape[0]:
            set_mocap_pose(data, mocap_id, pose)
        elif mocap_id < data.mocap_pos.shape[0]:
            set_mocap_pose(data, mocap_id, np.array([0, 0, -0.5, 1, 0, 0, 0]))


def set_state(plant: MujocoPlant, state: FloatTensor) -> None:
    """Set plant state."""
    plant.data.qpos = state[0 : plant.model.nq].cpu().numpy()
    plant.data.qvel = state[plant.model.nq :].cpu().numpy()


def visualize_state_or_trajectory(
    plant: MujocoPlant,
    time_step: float,
    state: FloatTensor | None = None,
    trajectory: FloatTensor | None = None,
    display_time: float = 5.0,
) -> None:
    """Visualize the system undergo a trajectory of states."""
    assert (state is not None) ^ (
        trajectory is not None
    ), "Requires either state or trajectory"

    model = plant.model
    data = plant.data
    if trajectory is not None:
        num_steps = trajectory.shape[0]
        horizon = (num_steps - 1) * time_step

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Close the viewer automatically after display_time wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < display_time:
            step_start = time.time()
            if trajectory is not None:
                # we perform zero-order hold interpolation
                theta = (time.time() - start) / horizon
                idx = min(num_steps - 1, int(round(theta * (num_steps - 1))))
                state = trajectory[idx]
            set_state(plant, state)
            mujoco.mj_step(model, data)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = time_step - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    viewer.close()
