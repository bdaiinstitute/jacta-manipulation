# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Callable, Tuple

import torch
from torch import FloatTensor

from jacta.planner.core.parameter_container import ParameterContainer
from jacta.planner.dynamics.mujoco_dynamics import MujocoPlant
from jacta.planner.dynamics.simulator_plant import SimulatorPlant


def finite_difference(
    f: Callable,
    x: FloatTensor,
    finite_diff_eps: float = 1e-4,
    centered: bool = True,
) -> FloatTensor:
    y = f(x)
    input_size = x.shape[0]
    output_size = y.shape[0]

    jacobian = torch.zeros((output_size, input_size))
    for i in range(input_size):
        eps = torch.zeros(input_size)
        eps[i] += finite_diff_eps
        if centered:
            jacobian[:, i] = (f(x + eps) - f(x - eps)) / (2 * finite_diff_eps)
        else:
            jacobian[:, i] = (f(x + eps) - y) / finite_diff_eps
    return jacobian


def rollout_plant(
    plant: SimulatorPlant, x0: FloatTensor, a0: FloatTensor, N: int
) -> Tuple[FloatTensor, FloatTensor]:
    plant.set_state(x0)
    plant.set_action(a0)
    actions = a0.repeat((2, 1))
    x1, _ = plant.dynamics(x0, actions, plant.sim_time_step * N)
    s1 = plant.get_sensor(x1)
    return x1, s1


def test_sub_stepped_gradients() -> None:
    finite_diff_eps = 1e-3
    params = ParameterContainer()
    params.load_base()
    params.model_filename = "box_push.xml"
    params.vis_filename = "box_push.yml"
    params.finite_diff_eps = finite_diff_eps
    params.autofill()

    plant = MujocoPlant(params)
    x0 = torch.tensor([3, 2, -1, 1.0])
    a0 = torch.tensor([3.5])

    for N in range(1, 20, 5):
        (
            state_gradient_state0,
            state_gradient_control0,
            sensor_gradient_state0,
            sensor_gradient_control0,
        ) = plant.get_sub_stepped_gradients(x0, a0, N)

        state_gradient_state1 = finite_difference(
            lambda x0: rollout_plant(plant, x0, a0, N)[0],  # noqa: B023
            x0,
            finite_diff_eps=finite_diff_eps,
            centered=True,
        )

        sensor_gradient_state1 = finite_difference(
            lambda x0: rollout_plant(plant, x0, a0, N)[1],  # noqa: B023
            x0,
            finite_diff_eps=finite_diff_eps,
            centered=True,
        )

        state_gradient_control1 = finite_difference(
            lambda a0: rollout_plant(plant, x0, a0, N)[0],  # noqa: B023
            a0,
            finite_diff_eps=finite_diff_eps,
            centered=True,
        )

        sensor_gradient_control1 = finite_difference(
            lambda a0: rollout_plant(plant, x0, a0, N)[1],  # noqa: B023
            a0,
            finite_diff_eps=finite_diff_eps,
            centered=True,
        )
        assert torch.norm(state_gradient_state0 - state_gradient_state1) < 1e-3
        assert torch.norm(sensor_gradient_state0 - sensor_gradient_state1) < 1e-3
        assert torch.norm(state_gradient_control0 - state_gradient_control1) < 1e-3
        assert torch.norm(sensor_gradient_control0 - sensor_gradient_control1) < 1e-3
