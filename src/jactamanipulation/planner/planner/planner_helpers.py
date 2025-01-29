# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
import numpy as np
import torch
from torch import BoolTensor, FloatTensor


def is_object_tilted(
    sensordata: FloatTensor,
    tilt_tolerance: float = np.pi / 9.0,
    y_axis_indexes: slice = slice(24, 27),  # noqa: B008
) -> BoolTensor:
    """Determine if an object is tilted beyond a specified tolerance.

    This function calculates the tilt angle of an object based on its y-axis orientation
    and compares it to a given tolerance to determine if the object is considered tilted.

    Args:
        sensordata: A tensor containing sensor data with `<frameyaxis>` sensors enabled.
        tilt_tolerance: The maximum allowable tilt angle in radians. Defaults to π/9 (20 degrees).
        y_axis_indexes: A slice object representing the indexes of the object's y-axis in the sensor data.

    Returns:
        bool: True if the object is tilted beyond the tolerance, False otherwise.
    """
    object_y_axis = sensordata[:, y_axis_indexes]

    # The vertical component is directly the z-component of the y-axis
    vertical_component = object_y_axis[:, 2]

    # The tilt angle is the angle between the object's y-axis and the vertical axis
    absolute_tilt_angle = torch.abs(torch.arcsin(vertical_component))

    # Check if the object is tilted
    is_tilted = absolute_tilt_angle > tilt_tolerance

    return is_tilted


def is_object_out_of_reach(
    sensordata: FloatTensor,
    reach_tolerance: float = 0.5,
    dist_indexes: slice = slice(21, 24),  # noqa: B008
) -> BoolTensor:
    """Determine if an object is out of reach based on its distance from the indexed distance sensor.

    Args:
        sensordata: A tensor containing sensor data with `<framepos>` sensor enabled.
        reach_tolerance: The maximum allowable distance in meters. Defaults to 1.5 meters.
        dist_indexes: A slice object representing the indexes of the object's distance in the sensor data.

    Returns:
        bool: True if the object is out of reach, False otherwise.
    """
    xyz_dist = sensordata[:, dist_indexes]

    # The distance from the indexed sensor to the object is the magnitude of the distance vector
    object_distance = torch.norm(xyz_dist, dim=-1)

    # Check if the object is out of reach
    is_out_of_reach = torch.abs(object_distance) > reach_tolerance

    return is_out_of_reach


def torso_too_close_to_object(
    sensordata: FloatTensor,
    torso_contact_indexes: slice = slice(0, 3),  # noqa: B008
    tolerance: float = 0.7,
) -> BoolTensor:
    """Determine if the robot's torso is too close to an object based on the distance from the indexed sensor.

    Args:
        sensordata: A tensor containing sensor data with `<framepos>` sensors enabled.
        torso_contact_indexes: A slice object representing the indexes of the robot's torso in the sensor data.
        tolerance: Allowable distance/radius in meters. Defaults to 0.7 meters.
    """
    xyz_dist = sensordata[:, torso_contact_indexes]

    distance = torch.norm(xyz_dist, dim=-1)

    is_in_contact = distance < tolerance

    return is_in_contact
