# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import warnings
from typing import Any, List

import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
from meshcat.visualizer import Visualizer

from dexterity.planner.visuals.quaternion_operations import (
    pose_to_transformation_matrix,
    quaternion_to_transformation_matrix,
)


def rgb_int_to_hex(rgb: List[int]) -> int:
    """Convert an RGB list of integers to a hexadecimal color value."""
    assert all(isinstance(c, int) and 0 <= c <= 255 for c in rgb)
    r, g, b = rgb
    return (r << 16) + (g << 8) + b


def rgb_float_to_hex(rgb: List[float]) -> int:
    """Convert an RGB list of floats to a hexadecimal color value."""
    assert all(0 <= c <= 1.0 for c in rgb)
    int_rgb = [int(c * 255) for c in rgb]
    return rgb_int_to_hex(int_rgb)


def rgba_overwrite(model_rgba: list[float], user_rgba: list[float] | float | None) -> list[float]:
    """Overwrite rgba values with user-defined values."""
    if user_rgba is None:  # we preserve mujoco color and opacity
        rgba = model_rgba
    elif isinstance(user_rgba, list):  # we overwrite color and opacity with user-provided values
        rgba = user_rgba
    else:  # we preserve color and overwrite opacity
        rgba = model_rgba[0:3] + [user_rgba]
    return rgba


class MeshPhysicalMaterial(g.GenericMaterial):
    """A material class with high reflectivity."""

    _type = "MeshPhysicalMaterial"


class MeshToonMaterial(g.GenericMaterial):
    """A material class for poster-like rendering."""

    _type = "MeshToonMaterial"


class SetPropertyCapital:
    """Class to set properties with keys that contain a capital letter. This bypasses a bug in Meshcat."""

    __slots__ = ["path", "key", "value"]

    def __init__(self, key: str, value: Any, path: str) -> None:
        self.key = key
        self.value = value
        self.path = path

    def lower(self) -> dict:
        """Lower

        Returns:
            dict: Lower
        """
        return {
            "type": "set_property",
            "path": self.path.lower(),
            "property": self.key,  # we don't apply .lower() on the key contrary to meshcat
            "value": self.value,
        }


def set_property_capital(visualizer: Visualizer, key: str, value: Any) -> None:
    """Set property for properties with keys that contain a capital letter."""
    return visualizer.window.send(SetPropertyCapital(key, value, visualizer.path))


Visualizer.set_property_capital = set_property_capital


def close_window(visualizer: Visualizer) -> None:
    if visualizer.window is not None:
        window = visualizer.window
        if window.zmq_socket:
            window.zmq_socket.close()
        if window.server_proc:
            window.server_proc.terminate()  # Terminate the server process
        window.context.term()  # Terminate the ZMQ context
        print("ViewerWindow closed and ZMQ server terminated.")


Visualizer.close_window = close_window


class Plane(g.Geometry):
    """Class that defines a Plane geometry.

    This is implemented in Meshcat, but it is not part of the last released version on PyPI
    """

    def __init__(
        self, width: float = 1.0, height: float = 1.0, widthSegments: float = 1.0, heightSegments: float = 1.0
    ) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.widthSegments = widthSegments
        self.heightSegments = heightSegments

    def lower(self, object_data: Any) -> dict:
        """Lower

        Args:
            object_data (Any): Object data

        Returns:
            dict: Lower
        """
        return {
            "uuid": self.uuid,
            "type": "PlaneGeometry",
            "width": self.width,
            "height": self.height,
            "widthSegments": self.widthSegments,
            "heightSegments": self.heightSegments,
        }


def set_color(visualizer: Visualizer, rgba: list[float]) -> None:
    """Set the color of all objects contained in the visualizer.

    Example:
        Set the color of the model "goal":
        set_color(visualizer["goal"], rgba)
    """
    visualizer.set_property("color", rgba)


def setup_visualizer(
    visualizer: Visualizer,
    axes_visible: bool = True,
    grid_visible: bool = True,
    zoom: float = 1.0,
    camera_pos: list[float] | None = None,
    top_color: list[float] | None = None,
    bottom_color: list[float] | None = None,
    negative_shadow: bool = False,
    positive_shadow: bool = True,
) -> None:
    """Set default properties of a visualizer including camera position, background color, shadows."""
    visualizer["/Cameras/default/rotated/<object>"].set_property("zoom", zoom)

    visualizer["/Lights/AmbientLight/<object>"].set_property("intensity", 0.6)
    visualizer["/Lights/FillLight/<object>"].set_property("intensity", 0.3)
    visualizer["/Lights/PointLightNegativeX/<object>"].set_property("intensity", 0.5)
    visualizer["/Lights/PointLightPositiveX/<object>"].set_property("intensity", 1.0)
    visualizer["/Lights/PointLightNegativeX/<object>"].set_property("distance", 20.0)
    visualizer["/Lights/PointLightPositiveX/<object>"].set_property("distance", 60.0)
    visualizer["/Lights/PointLightNegativeX/<object>"].set_property_capital("castShadow", negative_shadow)
    visualizer["/Lights/PointLightPositiveX/<object>"].set_property_capital("castShadow", positive_shadow)

    visualizer["/Grid"].set_property("visible", grid_visible)
    visualizer["/Axes"].set_property("visible", axes_visible)

    if camera_pos is not None:
        x, y, z = camera_pos
        transform = tf.translation_matrix([x, z, -y])
        visualizer["/Cameras/default/rotated/<object>"].set_transform(transform)

    colors = {"blue_green": [0.17, 0.67, 0.59], "blue": [0, 0.2, 0.63]}

    if top_color is None:
        top_color = colors["blue"]
    visualizer["/Background"].set_property("top_color", top_color)

    if bottom_color is None:
        bottom_color = colors["blue_green"]
    visualizer["/Background"].set_property("bottom_color", bottom_color)


def add_object(
    visualizer: Visualizer,
    name: str,
    obj: g.Geometry,
    transform: np.ndarray,
    material: g.Material | None = None,
) -> None:
    """Add a geometry object to the visualizer."""
    visualizer[name].set_object(obj, material)
    visualizer[name].set_transform(transform)


def add_ground(
    visualizer: Visualizer, height: float = 0.0, material: g.Material | None = None, name: str = "ground"
) -> None:
    """Add a ground plane to the visualizer."""
    obj = Plane(width=20.0, height=20.0)
    transform = tf.translation_matrix(np.array([0, 0, height]))
    add_object(visualizer, name, obj, transform, material)


def add_plane(
    visualizer: Visualizer,
    size: np.ndarray,
    pos: np.ndarray | None = None,
    quat: np.ndarray | None = None,
    material: g.Material | None = None,
    name: str = "plane",
) -> None:
    """Add a plane geometry to the visualizer with optional position, quaternion, material, and name."""
    width = float(size[0])
    height = float(size[1])
    obj = Plane(width=width, height=height)
    transform = pose_to_transformation_matrix(pos, quat)
    add_object(visualizer, name, obj, transform, material)


def add_sphere(
    visualizer: Visualizer,
    radius: float,
    pos: np.ndarray | None = None,
    quat: np.ndarray | None = None,
    material: g.Material | None = None,
    name: str = "sphere",
) -> None:
    """Add a sphere geometry to the visualizer with optional position, quaternion, material, and name."""
    radius = float(radius)
    obj = g.Sphere(radius)
    transform = pose_to_transformation_matrix(pos, quat)
    add_object(visualizer, name, obj, transform, material)


def add_cylinder(
    visualizer: Visualizer,
    radius: float,
    length: float,
    pos: np.ndarray | None = None,
    quat: np.ndarray | None = None,
    material: g.Material | None = None,
    name: str = "cylinder",
) -> None:
    """Add a cylinder geometry to the visualizer with optional position, quaternion, material, and name.

    The cylinder is aligned with the z-axis
    """
    obj = g.Cylinder(length, radius)
    transform = pose_to_transformation_matrix(pos, quat)
    # to align the cylinder centerline with the Z axis
    quat_offset = np.sqrt(2) / 2 * np.array([1, 1, 0, 0])
    rotation_offset = quaternion_to_transformation_matrix(quat_offset)
    transform = transform @ rotation_offset
    add_object(visualizer, name, obj, transform, material)


def add_box(
    visualizer: Visualizer,
    size: np.ndarray,
    pos: np.ndarray | None = None,
    quat: np.ndarray | None = None,
    material: g.Material | None = None,
    name: str = "box",
) -> None:
    """Add a box geometry to the visualizer with optional position, quaternion, material, and name."""
    size = np.array(size)
    obj = g.Box(size)
    transform = pose_to_transformation_matrix(pos, quat)
    add_object(visualizer, name, obj, transform, material)


def add_capsule(
    visualizer: Visualizer,
    radius: float,
    length: float,
    pos: np.ndarray | None = None,
    quat: np.ndarray | None = None,
    material: g.Material | None = None,
    name: str = "capsule",
) -> None:
    """Add a capsule geometry to the visualizer with optional position, quaternion, material, and name.

    The capsule is aligned with the z-axis
    """
    radius = float(radius)
    length = float(length)
    cylinder_obj = g.Cylinder(height=length, radius=radius)
    sphere_left_obj = g.Sphere(radius)
    sphere_right_obj = g.Sphere(radius)

    cylinder_transform = tf.rotation_matrix(np.pi / 2, [1, 0, 0.0])
    sphere_left_transform = tf.translation_matrix([0, 0, -length / 2])
    sphere_right_transform = tf.translation_matrix([0, 0, length / 2])

    add_object(visualizer, name + "/capsule/cylinder", cylinder_obj, cylinder_transform, material)
    add_object(visualizer, name + "/capsule/sphere_left", sphere_left_obj, sphere_left_transform, material)
    add_object(visualizer, name + "/capsule/sphere_right", sphere_right_obj, sphere_right_transform, material)

    transform = pose_to_transformation_matrix(pos, quat)
    visualizer[name + "/capsule"].set_transform(transform)


def add_ellipsoid(
    visualizer: Visualizer,
    covariance: np.ndarray,
    pos: np.ndarray | None = None,
    quat: np.ndarray | None = None,
    material: g.Material | None = None,
    name: str = "ellipsoid",
    confidence_interval_mode: bool = False,
) -> None:
    """Add an ellipsoid geometry to the visualizer

    Add an ellipsoid geometry to the visualizer based on the given covariance matrix,
    with optional position, quaternion, material, and name.

    The orthogonal matrix Q (eigenvectors):
        Q = eigenvectors

    The diagonal matrix sqrt_D (eigenvalues):
        D = np.diag(eigenvalues)

    covariance = Q @ D @ Q.T
    sqrt(covariance) = Q @ sqrt(D) @ Q.T

    With x in N(0, I) we get:
        y = mean + sqrt(covariance) @ x in N(mean, covariance)
    """
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if confidence_interval_mode:
        obj = g.Ellipsoid(3 * np.sqrt(eigenvalues))
    else:
        obj = g.Ellipsoid(eigenvalues)
    rotation = np.eye(4)
    rotation[0:3, 0:3] = eigenvectors
    transform = pose_to_transformation_matrix(pos, quat) @ rotation
    add_object(visualizer, name, obj, transform, material)


def add_mesh_by_name(
    visualizer: Visualizer,
    filename: str,
    pos: np.ndarray | None = None,
    quat: np.ndarray | None = None,
    material: g.Material | None = None,
    name: str = "mesh",
) -> None:
    """Add a mesh geometry to the visualizer from a file, with optional position, quaternion, material, and name."""
    obj = g.ObjMeshGeometry.from_file(filename)
    transform = pose_to_transformation_matrix(pos, quat)
    add_object(visualizer, name, obj, transform, material)


def add_mesh(
    visualizer: Visualizer,
    vertices: np.ndarray,
    faces: np.ndarray,
    pos: np.ndarray | None = None,
    quat: np.ndarray | None = None,
    material: g.Material | None = None,
    name: str = "mesh",
) -> None:
    """Add a triangular mesh geometry to the visualizer

    Add a triangular mesh geometry to the visualizer with specified vertices and faces,
    with optional position, quaternion, material, and name.

    Vertices: float (N, 3) and faces: int (M, 3).
    """
    obj = g.TriangularMeshGeometry(vertices, faces)
    transform = pose_to_transformation_matrix(pos, quat)
    add_object(visualizer, name, obj, transform, material)


def set_object(
    visualizer: Visualizer, pos: np.ndarray | None = None, quat: np.ndarray | None = None, name: str = "object"
) -> None:
    """Set the transformation (position and orientation) of an object in the visualizer."""
    if np.isnan(pos).any() or np.isnan(quat).any():
        pos = np.array([0, 0, -1.0])
        quat = np.array([1, 0, 0, 0.0])
        warnings.warn(f"NaN values in {name} transform. Setting to default values.", stacklevel=2)
    transform = pose_to_transformation_matrix(pos, quat)
    visualizer[name].set_transform(transform)
