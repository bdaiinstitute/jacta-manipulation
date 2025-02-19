# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import torch

from jacta.planner.core.parameter_container import ParameterContainer
from jacta.planner.dynamics.mujoco_dynamics import MujocoPlant


def my_sensor(x: torch.FloatTensor) -> torch.FloatTensor:
    return torch.tensor([x[1] - x[0], 0, 0.0])


def test_sensor_measurement() -> None:
    params = ParameterContainer()
    params.load_base()
    params.model_filename = "box_push.xml"
    params.vis_filename = "box_push.yml"
    params.finite_diff_eps = 1e-4
    params.autofill()

    plant = MujocoPlant(params)

    # when we get the sensor measurement we also set the state
    x0 = torch.tensor([3, -1, 1, -1.0])
    s0 = plant.get_sensor(x0)
    assert torch.allclose(s0, my_sensor(x0))
    x00 = plant.get_state()
    assert torch.allclose(x0, x00)
