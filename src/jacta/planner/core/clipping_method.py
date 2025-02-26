# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Optional

import torch
from torch import FloatTensor

from jacta.planner.core.parameter_container import ParameterContainer
from jacta.planner.core.types import ClippingType


def clip_actions(actions: FloatTensor, params: ParameterContainer) -> FloatTensor:
    match params.clipping_type:
        case ClippingType.CLIP:
            actions = torch.clamp(
                actions[..., :],
                min=params.action_bound_lower,
                max=params.action_bound_upper,
            )
        case ClippingType.SCALE:
            actions = box_scaling(
                actions[..., :], params.action_bound_lower, params.action_bound_upper
            )
        case _:
            raise ValueError(f"Unknown clipping method {params.clipping_type}")
    return actions


def box_scaling(
    v: FloatTensor,
    v_min: FloatTensor,
    v_max: FloatTensor,
    v_mid: Optional[FloatTensor] = None,
) -> FloatTensor:
    """Scales vector v down to ensure that the scaled version of v (v_bar) belongs to the box [v_min, v_max].
    The scaling is performed about a centerpoint v_mid.
    v = v_mid + n
    v_bar = v_mid + alpha n with alpha in [0, 1]
    """
    if not (torch.all(v_min < v_max)):
        raise ValueError("v_min must be less than v_max")

    if v_mid is None:
        v_mid = (v_min + v_max) / 2
    elif not (torch.all(v_mid >= v_min)) or not (torch.all(v_mid <= v_max)):
        raise ValueError("v_mid must be within v_min and v_max")

    alpha = 1.0
    n = v - v_mid

    alpha_up = 1 / torch.max(n / (v_max - v_mid))
    if alpha_up >= 0:
        alpha = min(alpha, torch.min(alpha_up).item())
    alpha_down = 1 / torch.max(n / (v_min - v_mid))
    if alpha_down >= 0:
        alpha = min(alpha, torch.min(alpha_down).item())

    v_bar = v_mid + alpha * n
    return v_bar
