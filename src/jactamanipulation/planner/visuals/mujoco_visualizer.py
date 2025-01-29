# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

"""Class for rendering Mujoco trajectories in meshcat"""

import os
from typing import Any, Dict, List, Optional, TypeAlias

import matplotlib.pyplot as plt
import meshcat.transformations as tf
import numpy as np
import torch
from meshcat.animation import Animation
from meshcat.visualizer import Visualizer
from mujoco import MjData, MjModel, mj_kinematics

from jactamanipulation.planner.dynamics.mujoco_dynamics import MujocoPlant
from jactamanipulation.planner.dynamics.mujoco_utils import (
    get_body_name,
    get_geom_name,
    get_geometry_rgba,
    get_mesh_data,
)
from jactamanipulation.planner.visuals.meshcat_visualizer import (
    MeshPhysicalMaterial,
    add_box,
    add_capsule,
    add_cylinder,
    add_ellipsoid,
    add_mesh,
    add_plane,
    add_sphere,
    rgb_float_to_hex,
    rgba_overwrite,
    set_object,
    setup_visualizer,
)

DEFAULT_COLORS = {
    "trajectory": None,
    "goal": [0.2, 0.2, 0.6, 0.2],
}


Trajectory: TypeAlias = torch.FloatTensor | np.ndarray
RGBA: TypeAlias = Optional[List[float] | float]


class MujocoRenderer:
    """MujocoMeshcatRenderer is a class for rendering the Mujoco trajectories in meshcat.

    It parses bodies from Mujoco.MjModel and animates them using minimal coordinates.

    Example:
    ```
    >>> renderer = MujocoRenderer(plant = plant, time_step=0.01)
    >>> for state in state_trajectory:
    >>>     renderer.render(state)
    >>> renderer.show_goal(goal_state)
    >>> renderer.play()
    >>> OR
    >>> renderer.save("recording.html")
    """

    def __init__(
        self,
        *,
        plant: Optional[MujocoPlant] = None,
        model_filepath: Optional[str] = None,
        time_step: Optional[float] = None,
        collision_geometry_opacity: Optional[float] = None,
    ) -> None:
        self.plant = plant
        if self.plant is not None:
            self.mj_model = self.plant.model
            self.data = self.plant.data
            self.time_step = self.plant.sim_time_step
            self.collision_geometry_opacity = collision_geometry_opacity
        else:
            if model_filepath is None:
                raise ValueError("Either a plant or a model filepath must be provided!")
            self.mj_model = MjModel.from_xml_path(str(model_filepath))
            self.data = MjData(self.mj_model)
            self.time_step = time_step
            self.collision_geometry_opacity = collision_geometry_opacity
        if self.time_step is None:
            self.time_step = self.mj_model.opt.timestep
        self.visualizer = Visualizer()
        self.trajectory_names: list = []
        setup_visualizer(self.visualizer)
        self.reset()
        self.markers: dict = {}

    @property
    def framerate(self) -> int:
        """Framerate

        Returns:
            int: Framerate
        """
        assert self.time_step is not None
        return int(np.floor(1 / self.time_step))

    def load_model(self, trajectory_name: str = "trajectory", rgba: RGBA = None, force_reload: bool = True) -> None:
        """Load model visualization for a given trajectoryname if not already loaded."""
        if trajectory_name in self.trajectory_names and not force_reload:
            return
        self.trajectory_names.append(trajectory_name)

        model = self.mj_model

        num_geoms = model.ngeom
        for i in range(num_geoms):
            geom_type = model.geom_type[i]
            geom_pos = model.geom_pos[i]
            geom_quat = model.geom_quat[i]
            geom_bodyid = model.geom_bodyid[i]
            geom_size = model.geom_size[i]
            # group 3 is reserved for collision geometries, they are typically turned off for visualization
            geom_group = model.geom_group[i]

            body_name = get_body_name(model, geom_bodyid).replace("/", "_")
            mujoco_rgba = get_geometry_rgba(model, i)
            merged_rgba = rgba_overwrite(mujoco_rgba, rgba)

            hex_color = rgb_float_to_hex(merged_rgba[0:3])
            opacity = float(merged_rgba[3])
            if geom_group == 3 and self.collision_geometry_opacity is not None:
                opacity = self.collision_geometry_opacity
            material = MeshPhysicalMaterial(color=hex_color, opacity=opacity, roughness=0.1, metalness=0.1)

            body_name = get_body_name(model, geom_bodyid).replace("/", "_")
            geom_name = get_geom_name(model, i)
            visualizer_body = self.visualizer[trajectory_name][body_name]

            # geometry types: plane, hfield, sphere, capsule, ellipsoid, cylinder, box, mesh, sdf
            match geom_type:
                case 0:
                    size = 2 * geom_size[0:2]
                    add_plane(visualizer_body, size, geom_pos, geom_quat, material=material, name=geom_name)
                case 1:
                    raise NotImplementedError("hfield geometry type (case 1) is not implemented.")
                case 2:
                    radius = geom_size[0]
                    add_sphere(visualizer_body, radius, geom_pos, geom_quat, material=material, name=geom_name)
                case 3:
                    radius = geom_size[0]
                    length = 2 * geom_size[1]
                    add_capsule(visualizer_body, radius, length, geom_pos, geom_quat, material=material, name=geom_name)
                case 4:
                    covariance = np.diag(geom_size)
                    add_ellipsoid(visualizer_body, covariance, geom_pos, geom_quat, material=material, name=geom_name)
                case 5:
                    radius = geom_size[0]
                    length = 2 * geom_size[1]
                    add_cylinder(
                        visualizer_body, radius, length, geom_pos, geom_quat, material=material, name=geom_name
                    )
                case 6:
                    size = 2 * geom_size
                    add_box(visualizer_body, size, pos=geom_pos, quat=geom_quat, material=material, name=geom_name)
                case 7:
                    meshid = model.geom_dataid[i]
                    vertices, faces = get_mesh_data(model, meshid)
                    add_mesh(visualizer_body, vertices, faces, geom_pos, geom_quat, material=material, name=geom_name)
                case 8:
                    raise NotImplementedError("sdf geometry type (case 8) is not implemented.")

    def set_model(
        self, visualizer: Visualizer, joint_position: np.ndarray, trajectory_name: str = "trajectory"
    ) -> None:
        """Set model visualization based on given joint positions for a specific trajectory."""
        model = self.mj_model
        data = self.data
        num_bodies = model.nbody

        # forward kinematics
        data.qpos = joint_position
        data.qvel = np.zeros(model.nv)
        mj_kinematics(model, data)

        for i in range(num_bodies):
            body_name = get_body_name(model, i).replace("/", "_")
            body_pos = data.xpos[i]
            body_quat = data.xquat[i]
            set_object(visualizer[trajectory_name], body_pos, body_quat, name=body_name)

    def reset(self) -> None:
        """Reset the visualization environment by deleting all trajectories and animations."""
        self.frame_index = 0
        for trajectory_name in self.trajectory_names:
            self.visualizer[trajectory_name].delete()
        self.trajectory_names = []
        self.animation = Animation(default_framerate=self.framerate)

    def initialize_markers(self, marker_info: Dict[str, Dict[str, Any]]) -> None:
        """Initialize multiple markers at once.

        Args:
            marker_info: A dictionary where keys are marker names and values are
                         dictionaries containing 'pos', 'rgba', and 'radius' for each marker.
        """
        for name, info in marker_info.items():
            self.add_marker(name, info["pos"], info["rgba"], info["radius"])
            self.markers[name] = info

    def update_markers(self, marker_poses: Dict[str, np.ndarray]) -> None:
        """Update the positions of multiple markers at once.

        Args:
            marker_poses: A dictionary where keys are marker names and values are
                          the new positions (np.ndarray) for each marker.
        """
        for name, pos in marker_poses.items():
            self.animate_marker(name, pos)

    def add_marker(
        self,
        name: str,
        pos: np.ndarray,
        rgba: list[float],
        radius: float = 0.02,
    ) -> None:
        """Add a marker to the visualization."""
        hex_color = rgb_float_to_hex(rgba[0:3])
        opacity = rgba[3] if isinstance(rgba, list) else rgba
        material = MeshPhysicalMaterial(color=hex_color, opacity=opacity, roughness=0.1, metalness=0.1)
        add_sphere(
            visualizer=self.visualizer["markers"],
            radius=radius,
            pos=pos,
            quat=np.array([1, 0, 0, 0]),
            material=material,
            name=name,
        )

    def set_marker(self, visualizer: Visualizer, pos: np.ndarray, name: str) -> None:
        """Set the position of a marker in the visualization."""
        set_object(visualizer["markers"], pos, quat=np.array([1, 0, 0, 0]), name=name)

    def animate_marker(self, name: str, pos: np.ndarray) -> None:
        """Animate a marker in the visualization.

        Note:
            This function should be called after the .render() function to animate the marker.
        """
        with self.animation.at_frame(self.visualizer, self.frame_index) as frame:
            self.set_marker(frame, pos, name)

    def render(
        self,
        trajectory: Trajectory,
        frame_index: int = -1,
        trajectory_name: str = "trajectory",
        rgba: RGBA = None,
    ) -> None:
        """Render the model at a specific frame using joint positions and trajectory name."""
        if frame_index == -1:  # if frame_index is not provided use internal frame counter
            frame_index = self.frame_index
            self.frame_index += 1
        self.load_model(trajectory_name=trajectory_name, rgba=rgba, force_reload=False)
        joint_positions = self.extract_joint_positions(trajectory)
        with self.animation.at_frame(self.visualizer, frame_index) as frame:
            self.set_model(frame, joint_positions, trajectory_name=trajectory_name)

    def init_points(self, N_points: int, name: str, color: tuple = (0, 1, 0)) -> None:
        """Trajectory visualization utils"""
        c = rgb_float_to_hex(color)
        material = MeshPhysicalMaterial(color=c, opacity=0.8, roughness=0.1, metalness=0.01)
        for i in range(N_points):
            add_sphere(
                self.visualizer,
                0.03,
                pos=np.zeros(3),
                quat=np.array([1, 0, 0, 0]),
                material=material,
                name=f"{name}_{i}",
            )

    def update_points(self, points: np.ndarray, name: str) -> None:
        """Trajectory visualization utils"""
        with self.animation.at_frame(self.visualizer, self.frame_index) as frame:
            for i in range(points.shape[0]):
                pos = points[i]
                frame[f"{name}_{i}"].set_transform(tf.translation_matrix(pos))

    def initialize_candidate_trajectories(self, N_candidates: int, N_eval: int, part: str = "base") -> None:
        """Initialize candidate trajectories.

        Initialize candidate trajectories for visualization of spline candidate trajectories for either
        the base or the arm. Trajectories are visualized arrays of points in 3D space.
        """
        cmap = plt.get_cmap("bwr")
        colors = [rgb_float_to_hex(np.array(cmap(i)[:3])) for i in np.linspace(0, 1, N_candidates)]
        # show best candidate in yellow
        colors[0] = rgb_float_to_hex([1, 1, 0])

        for i in range(N_candidates):
            material = MeshPhysicalMaterial(color=colors[i], opacity=0.8, roughness=0.1, metalness=0.01)
            for j in range(N_eval):
                add_sphere(
                    self.visualizer[f"{part}_cands"][f"cand_{i}"],
                    0.01,
                    pos=np.zeros(3),
                    quat=np.array([1, 0, 0, 0]),
                    material=material,
                    name=f"pt_{j}",
                )

    def update_candidate_trajectories(self, trajectories: List[np.ndarray], part: str = "base") -> None:
        """Trajectory visualization utils"""
        with self.animation.at_frame(self.visualizer, self.frame_index) as frame:
            for i in range(len(trajectories)):
                for j in range(trajectories[i].shape[0]):
                    pos = trajectories[i][j, :3]
                    if part == "base":
                        pos[2] = trajectories[i][j, -1]
                    transform = tf.translation_matrix(pos)
                    frame[f"{part}_cands"][f"cand_{i}"][f"pt_{j}"].set_transform(transform)

    def init_ee_position(self) -> None:
        """Trajectory visualization utils"""
        # add point for EE position
        c = rgb_float_to_hex([0, 1, 0])
        material = MeshPhysicalMaterial(color=c, opacity=0.8, roughness=0.1, metalness=0.01)
        add_sphere(
            self.visualizer, 0.05, pos=np.zeros(3), quat=np.array([1, 0, 0, 0]), material=material, name="ee_pos"
        )

    def update_ee_position(self, pos: np.ndarray) -> None:
        """Trajectory visualization utils"""
        with self.animation.at_frame(self.visualizer, self.frame_index) as frame:
            T = np.eye(4)
            T[:3, 3] = pos
            frame["ee_pos"].set_transform(T)

    def show_trajectory(
        self,
        joint_positions: np.ndarray,
        trajectory_name: str = "trajectory",
        rgba: RGBA = None,
        force_reload: bool = True,
    ) -> None:
        """Render a trajectory of joint positions with optional RGBA color."""
        num_frames = joint_positions.shape[0]
        self.load_model(trajectory_name=trajectory_name, rgba=rgba, force_reload=force_reload)
        for i in range(num_frames):
            with self.animation.at_frame(self.visualizer, i) as frame:
                self.set_model(frame, joint_positions[i, :], trajectory_name=trajectory_name)

    def show_goal(
        self,
        joint_position: Trajectory,
        goal_name: str = "goal",
        rgba: RGBA = None,
        force_reload: bool = True,
    ) -> None:
        """Show the goal joint positions with optional RGBA color."""
        self.load_model(trajectory_name=goal_name, rgba=rgba, force_reload=force_reload)
        with self.animation.at_frame(self.visualizer, 0) as frame:
            self.set_model(frame, joint_position, trajectory_name=goal_name)

    def extract_joint_positions(self, trajectory: Trajectory) -> np.ndarray:
        """Extracts joint positions as a numpy array.

        This method takes a trajectory, which can be either a torch.FloatTensor or a numpy.ndarray,
        and extracts the joint positions up to the number of joints (nq) in the plant model.

        Args:
            trajectory (Trajectory): The input trajectory containing joint positions.

        Returns:
            np.ndarray: joint positions
        """
        nq = self.mj_model.nq
        if isinstance(trajectory, torch.Tensor):
            joint_positions = trajectory.cpu().numpy()[..., 0:nq]
        else:
            joint_positions = trajectory[..., 0:nq]
        return joint_positions

    def show_box_goal(self, pos: np.ndarray, quat: np.ndarray, size: np.ndarray) -> None:
        """Show the goal object position and orientation."""
        mat = MeshPhysicalMaterial(color=rgb_float_to_hex([1, 0.2, 0.6]), opacity=0.5, roughness=0.1, metalness=0.1)
        with self.animation.at_frame(self.visualizer, 0) as frame:  # noqa: F841
            add_box(self.visualizer, size, pos=pos, quat=quat, material=mat, name="goal")

    def show(
        self,
        trajectory: Trajectory,
        goal: Optional[Trajectory] = None,
        colors: Dict[str, Any] = DEFAULT_COLORS,
    ) -> None:
        """Visualize a trajectory alongside a goal pose for the robot.

        Args:
            trajectory: sequence of states (num_frames x nx) OR configurations (num_frames x nq)
            goal: sequence of states (num_frames x nx) OR configurations (num_frames x nq)
            colors: dictionary of {trajectory_name : colors} a color can be specified with
                a list (rgba), a float (opacity, original colors are preserved), None (original colors
                and opacity are preserved). E.g.
                DEFAULT_COLORS = {
                    "trajectory": 0.6,
                    "goal": [0.2, 0.2, 0.6, 0.2],
                }
        """
        self.reset()
        joint_positions = self.extract_joint_positions(trajectory)
        self.show_trajectory(joint_positions, trajectory_name="trajectory", rgba=colors["trajectory"])

        if goal is not None:
            goal_joint_positions = self.extract_joint_positions(goal)
            self.show_goal(goal_joint_positions, goal_name="goal", rgba=colors["goal"])
        self.play(wait_for_input=False)

    def play(self, wait_for_input: bool = False) -> None:
        """Play the animation."""
        self.visualizer.set_animation(self.animation)
        if wait_for_input:
            self.wait_for_input()

    def wait_for_input(self, message: str = "Press Enter to continue...") -> None:
        """Waits for user input

        Args:
            message (str, optional): Input message. Defaults to "Press Enter to continue...".
        """
        input(message)

    def get_html(self) -> str:
        """Generate static HTML representation of the current visualization."""
        self.play(False)
        return self.visualizer.static_html()

    def save(self, filename: str = "dexterity/meshcat_recording.html") -> None:
        """Save the current visualization as an HTML file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        html = self.get_html()
        with open(filename, "w") as f:
            f.write(html)
