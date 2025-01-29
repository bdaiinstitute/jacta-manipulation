# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
import numpy as np

from dexterity.planner.scenes import spot_sensors
from dexterity.spot_hardware.low_level.constants import ARM_UNSTOWED_POS


def slice_union(*slices: slice) -> slice:
    """Create a union of multiple slices, ensuring they are contiguous."""
    # Sort slices based on their start values
    sorted_slices = sorted(slices, key=lambda s: s.start)

    start = sorted_slices[0].start
    stop = sorted_slices[-1].stop

    # Check for contiguity
    for i in range(len(sorted_slices) - 1):
        if sorted_slices[i].stop != sorted_slices[i + 1].start:
            raise ValueError("Slices are not contiguous")

    return slice(start, stop)


# TODO(maks):
# the class below should be generated from the mujoco model
# for now it assumes that there is 1 spot and 1 object (in that order)


class StateArray:
    """State Array for ThreadedLocomotionPlant"""

    # Position slices
    base_pos = slice(0, 3)
    base_quat = slice(3, 7)
    fl_leg = slice(7, 10)
    fr_leg = slice(10, 13)
    hl_leg = slice(13, 16)
    hr_leg = slice(16, 19)
    arm = slice(19, 26)
    object_pos = slice(26, 29)
    object_quat = slice(29, 33)

    # Velocity slices
    base_lin_vel = slice(33, 36)
    base_ang_vel = slice(36, 39)
    fl_leg_vel = slice(39, 42)
    fr_leg_vel = slice(42, 45)
    hl_leg_vel = slice(45, 48)
    hr_leg_vel = slice(48, 51)
    arm_vel = slice(51, 58)
    object_lin_vel = slice(58, 61)
    object_ang_vel = slice(61, 64)

    # Unions
    legs = slice_union(fl_leg, fr_leg, hl_leg, hr_leg)
    legs_vel = slice_union(fl_leg_vel, fr_leg_vel, hl_leg_vel, hr_leg_vel)

    size = 64


class CommandArray:
    """Command Array for ThreadedLocomotionPlant"""

    longitudinal_vel = slice(0, 1)
    lateral_vel = slice(1, 2)
    yaw_vel = slice(2, 3)
    arm = slice(3, 10)
    fl_leg = slice(10, 13)
    fr_leg = slice(13, 16)
    hl_leg = slice(16, 19)
    hr_leg = slice(19, 22)
    base_roll = slice(22, 23)
    base_pitch = slice(23, 24)
    base_height = slice(24, 25)

    size: int = 25

    legs = slice_union(fl_leg, fr_leg, hl_leg, hr_leg)
    velocities = slice_union(longitudinal_vel, lateral_vel, yaw_vel)
    non_velocities = slice_union(arm, legs, base_roll, base_pitch, base_height)

    @classmethod
    def create(cls, num_systems: int) -> np.ndarray:
        """Create a new command array with default values."""
        commands = np.zeros((num_systems, cls.size))
        commands = cls.set_defaults(commands)
        return commands

    @classmethod
    def set_defaults(cls, commands: np.ndarray) -> np.ndarray:
        """Set default values for the command array."""
        commands[:, cls.base_height] = 0.55
        commands[:, cls.arm] = ARM_UNSTOWED_POS
        return commands

    @classmethod
    def update(cls, commands: np.ndarray, **kwargs: dict) -> np.ndarray:
        """Update the command array with new values."""
        for key, value in kwargs.items():
            if hasattr(cls, key):
                commands[:, getattr(cls, key)] = value
            else:
                raise ValueError(f"Invalid field name: {key}")
        return commands

    @classmethod
    def as_dict(cls, command: np.ndarray) -> dict:
        """Convert a command array to a dictionary."""
        return {
            name: command[getattr(cls, name)].flatten() for name in vars(cls) if isinstance(getattr(cls, name), slice)
        }


class SensorArray:
    """Sensor Array for ThreadedLocomotionPlant (spot + object)"""

    # Position slices
    body_object = slice(0, 3)
    object_y_axis = slice(3, 6)
    fngr_object = slice(6, 9)
    trace_body = slice(9, 12)
    trace_gripper = slice(12, 15)

    def __init__(self) -> None:
        if len(spot_sensors) != 5:
            raise ValueError(
                "Number of sensors in dexterity.planner.scenes.spot_sensors does not match SensorArray. "
                "Please adjust the SensorArray accordingly."
            )
