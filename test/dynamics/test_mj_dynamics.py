# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import torch
from torch import FloatTensor

from jacta.planner.core.parameter_container import ParameterContainer
from jacta.planner.dynamics.mujoco_dynamics import MujocoPlant

torch.manual_seed(0)


def random_quaternion() -> FloatTensor:
    v = torch.rand(4)
    v = v / torch.norm(v)
    return v


def normalize_vector(v: FloatTensor) -> FloatTensor:
    norm_v = torch.norm(v)
    if norm_v != 0:
        return v / norm_v
    else:
        return v


def mj_state_algebra(
    plant: MujocoPlant, state: FloatTensor, actions: FloatTensor
) -> None:
    nq = plant.model.nq
    nv = plant.model.nv
    na = plant.model.na
    time_step = plant.model.opt.timestep

    s0 = state
    s1, _ = plant.dynamics(s0, actions, plant.sim_time_step)
    ds = plant.state_difference(s0, s1, time_step)
    s1_ = plant.state_addition(s0, ds, time_step)

    # dimensions are correct
    assert len(ds) == 2 * nv + na
    assert len(s0) == nq + nv + na
    assert len(s1) == nq + nv + na
    assert len(s1_) == nq + nv + na

    # ds is of the right sign for the 3 first coordinates (positions x, y, z)
    assert torch.allclose(ds[0:3], (s1[0:3] - s0[0:3]) / time_step)

    # we recover the same state after the difference and addition operations
    assert torch.allclose(s1_, s1)

    # Additional tests with different time steps
    for dt in [1.0, 0.343, -0.343]:
        ds = plant.state_difference(s0, s1, dt)
        s1_ = plant.state_addition(s0, ds, dt)
        assert torch.allclose(s1_, s1)


def decompose_state_dimensions_utils(model_filename: str) -> None:
    params = ParameterContainer()
    params.load_base()
    params.model_filename = model_filename
    params.autofill()
    plant = MujocoPlant(params)

    nq = plant.model.nq
    nv = plant.model.nv

    # assert that all elements of the array are in the list
    assert all(elem in range(nq) for elem in plant.actuated_pos)
    assert all(elem in range(nq) for elem in plant.unactuated_pos)
    # assert that the pos dimensions are all accounted for
    assert len(plant.actuated_pos) + len(plant.unactuated_pos) == nq
    pos = torch.cat((plant.actuated_pos, plant.unactuated_pos))
    assert set(pos.cpu().numpy()) == set(range(nq))

    # assert that all elements of the array are in the list
    assert all(elem in range(nq, nq + nv) for elem in plant.actuated_vel)
    assert all(elem in range(nq, nq + nv) for elem in plant.unactuated_vel)
    # assert that the vel dimensions are all accounted for
    assert len(plant.actuated_vel) + len(plant.unactuated_vel) == nv
    vel = torch.cat((plant.actuated_vel, plant.unactuated_vel))
    assert set(vel.cpu().numpy()) == set(range(nq, nq + nv))

    # assert that the state dimensions are all accounted for
    assert len(plant.actuated_state) + len(plant.unactuated_state) == nq + nv
    state = torch.cat((plant.actuated_state, plant.unactuated_state))
    assert set(state.cpu().numpy()) == set(range(nq + nv))


def test_decompose_state_dimensions() -> None:
    decompose_state_dimensions_utils("box_push.xml")
    decompose_state_dimensions_utils("box_push_2d.xml")
    decompose_state_dimensions_utils("planar_hand.xml")
    decompose_state_dimensions_utils("floating_hand.xml")
    decompose_state_dimensions_utils("allegro_scene_tilt_wrist.xml")
    decompose_state_dimensions_utils("spot_standing_box.xml")
    decompose_state_dimensions_utils("spot_bimanual_stool.xml")
    decompose_state_dimensions_utils("spot_lying_ball.xml")
