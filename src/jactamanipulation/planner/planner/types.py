# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
import warnings
from enum import Enum
from typing import Any, Optional

import torch


class ClippingType(Enum):
    """Action clipping type"""

    CLIP = 0
    SCALE = 1

    def __str__(self) -> str:
        return f"{self.name}"


class SelectionType(Enum):
    """Node selection type"""

    PARETO = 0
    LAST = 1

    def __str__(self) -> str:
        return f"{self.name}"


class ActionType(Enum):
    """Action type

    - ranged corresponds to random action (gradient free),
    - proximity corresponds to optimal action to get close to the object of interest (gradient based),
    - continuation corresponds to action continuing the previous motion (gradient free),
    - goal corresponds to optimal action to get to the goal state (gradient based),
    - expert corresponds to action exploiting domain knowledge,
    """

    RANGED = 0
    PROXIMITY = 1
    CONTINUATION = 2
    GOAL = 3
    EXPERT = 4
    NON_EXPERT = 5

    def __str__(self) -> str:
        return f"{self.name}"


ACTION_TYPE_DIRECTIONAL_MAP = {
    ActionType.RANGED: True,
    ActionType.PROXIMITY: True,
    ActionType.CONTINUATION: True,
    ActionType.GOAL: True,
    ActionType.EXPERT: False,
    ActionType.NON_EXPERT: True,
}


class ActionMode(Enum):
    """Action Mode"""

    # for start action: this is the current state
    # for end action: this is the current state + the relative action
    RELATIVE_TO_CURRENT_STATE = 0
    # for start action: this is the previous end action
    # for end action: this is the previous end action + the relative action
    RELATIVE_TO_PREVIOUS_END_ACTION = 1
    # for start action: not implemented
    # for end action: this is the absolute action
    ABSOLUTE_ACTION = 2

    def __str__(self) -> str:
        return f"{self.name}"


class ControlType(Enum):
    """Type of interpolation applied for interpolating between start and end actions"""

    ZERO_ORDER_HOLD = 0
    FIRST_ORDER_HOLD = 1

    def __str__(self) -> str:
        return f"{self.name}"


def set_default_device_and_dtype(device: Optional[str] = None, dtype: torch.dtype = torch.float32) -> None:
    """Set default device and dtype for torch tensors."""
    torch.set_default_dtype(dtype)

    if device is not None:
        torch.set_default_device(device)
    elif torch.cuda.is_available():
        torch.set_default_device("cuda:0")
    else:
        warnings.warn("No CUDA device found. Defaulting to CPU.", stacklevel=2)
        torch.set_default_device("cpu")


def convert_dtype(attr: Any, dtype: Optional[torch.dtype] = None, int_dtype: torch.dtype = torch.int64) -> Any:
    """Cast list and tensors to user-specified data type for floating point numbers and integers."""
    if dtype is None:
        dtype = torch.get_default_dtype()

    if isinstance(attr, list) and all(isinstance(x, (float, int)) for x in attr):
        return torch.tensor(attr, dtype=dtype)
    if isinstance(attr, torch.Tensor):
        if torch.is_floating_point(attr):
            return attr.to(dtype)
        elif not torch.is_complex(attr) and attr.dtype != torch.bool:
            return attr.to(int_dtype)
    return attr
