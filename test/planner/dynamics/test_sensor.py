# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import torch

from dexterity.planner.dynamics.mujoco_dynamics import MujocoPlant
from dexterity.planner.planner.parameter_container import ParameterContainer


def my_sensor(x: torch.FloatTensor) -> torch.FloatTensor:
    return torch.tensor([x[1] - x[0], 0, 0.0])


def test_sensor_measurement() -> None:
    params = ParameterContainer()
    params.load_base()
    params.model_filename = "dexterity/models/xml/scenes/legacy/box_push.xml"
    params.finite_diff_eps = 1e-4
    params.autofill()

    plant = MujocoPlant(params=params)

    # when we get the sensor measurement we also set the state
    x0 = torch.tensor([3, -1, 1, -1.0])
    s0 = plant.get_sensor(x0)
    assert torch.allclose(s0, my_sensor(x0))
    x00 = plant.get_state()
    assert torch.allclose(x0, x00)
