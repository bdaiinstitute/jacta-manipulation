# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
from dataclasses import dataclass

import numpy as np

# Standard joint positions
LEGS_SITTING_POS = np.array(
    [
        0.48,
        1.26,
        -2.7929,
        -0.48,
        1.26,
        -2.7929,
        0.48,
        1.26,
        -2.7929,
        -0.48,
        1.26,
        -2.7929,
    ]
)

LEGS_STANDING_POS = np.array(
    [
        0.12,
        0.72,
        -1.45,
        -0.12,
        0.72,
        -1.45,
        0.12,
        0.72,
        -1.45,
        -0.12,
        0.72,
        -1.45,
    ]
)

ARM_STOWED_POS = np.array(
    [
        0,
        -3.11,
        3.13,
        1.56,
        0,
        -1.56,
        0,
    ]
)

ARM_UNSTOWED_POS = np.array(
    [
        0,
        -0.9,
        1.8,
        0,
        -0.9,
        0,
        0,
    ]
)

SITTING_STOWED_POS = np.concatenate((LEGS_SITTING_POS, ARM_STOWED_POS))

SITTING_UNSTOWED_POS = np.concatenate((LEGS_SITTING_POS, ARM_UNSTOWED_POS))

STANDING_STOWED_POS = np.concatenate((LEGS_STANDING_POS, ARM_STOWED_POS))

STANDING_UNSTOWED_POS = np.concatenate((LEGS_STANDING_POS, ARM_UNSTOWED_POS))

STANDING_HEIGHT = 0.52


ARM_LINK_NAMES = [
    "arm_link_sh0",
    "arm_link_sh1",
    "arm_link_el0",
    "arm_link_el1",
    "arm_link_wr0",
    "arm_link_wr1",
    "arm_link_fngr",
]

LEG_LINK_NAMES = [
    "front_left_hip",
    "front_left_upper_leg",
    "front_left_lower_leg",
    "front_right_hip",
    "front_right_upper_leg",
    "front_right_lower_leg",
    "rear_left_hip",
    "rear_left_upper_leg",
    "rear_left_lower_leg",
    "rear_right_hip",
    "rear_right_upper_leg",
    "rear_right_lower_leg",
]

ARM_JOINT_NAMES = [
    "arm_sh0",
    "arm_sh1",
    "arm_el0",
    "arm_el1",
    "arm_wr0",
    "arm_wr1",
    "arm_f1x",
]

LEG_JOINT_NAMES = [
    "joint_front_left_hip_x",
    "joint_front_left_hip_y",
    "joint_front_left_knee",
    "joint_front_right_hip_x",
    "joint_front_right_hip_y",
    "joint_front_right_knee",
    "joint_rear_left_hip_x",
    "joint_rear_left_hip_y",
    "joint_rear_left_knee",
    "joint_rear_right_hip_x",
    "joint_rear_right_hip_y",
    "joint_rear_right_knee",
]

SPOT_BODY_SLICE: slice = slice(3)
SPOT_QUAT_SLICE: slice = slice(3, 7)
SPOT_LEGS_SLICE: slice = slice(7, 19)
SPOT_ARMS_SLICE: slice = slice(19, 26)

SPOT_BODY_VEL_SLICE: slice = slice(26, 29)
SPOT_ANGS_VEL_SLICE: slice = slice(29, 32)
SPOT_LEGS_VEL_SLICE: slice = slice(32, 44)
SPOT_ARMS_VEL_SLICE: slice = slice(44, 51)


@dataclass
class SPOT_STATE_INDS:
    """Spot's indices for where relevant body parts are"""

    body_slice: slice = SPOT_BODY_SLICE
    quat_slice: slice = SPOT_QUAT_SLICE
    legs_slice: slice = SPOT_LEGS_SLICE
    arms_slice: slice = SPOT_ARMS_SLICE

    body_vel_slice: slice = SPOT_BODY_VEL_SLICE
    angs_vel_slice: slice = SPOT_ANGS_VEL_SLICE
    legs_vel_slice: slice = SPOT_LEGS_VEL_SLICE
    arms_vel_slice: slice = SPOT_ARMS_VEL_SLICE
