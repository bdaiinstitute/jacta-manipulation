# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
import torch
from planner_setup import planner_setup
from torch import tensor

from jacta.planner.core.parameter_container import ParameterContainer


def test_random_actions() -> None:
    params = ParameterContainer()
    params.parse_params("planar_hand", "test")
    planner = planner_setup(params, True)
    sorted_ids = planner.graph.sorted_progress_ids(params.reward_based)
    time_step = params.action_time_step
    for _ in range(100):
        idx = sorted_ids[torch.randint(low=0, high=len(sorted_ids), size=()).item()]
        directions = planner.action_sampler.random_directions(tensor([idx]))
        action = planner.action_sampler.directions_actions(tensor([idx]), directions)[0]
        assert torch.any(directions != 0)  # check non-zero
        assert torch.all(-params.action_range * time_step <= action)
        assert torch.all(action <= params.action_range * time_step)


def test_proximity_actions() -> None:
    params = ParameterContainer()
    params.parse_params("box_push", "test")
    planner = planner_setup(params, True)
    sorted_ids = planner.graph.sorted_progress_ids(params.reward_based)
    time_step = params.action_time_step
    for _ in range(100):
        idx = sorted_ids[torch.randint(low=0, high=len(sorted_ids), size=()).item()]
        directions = planner.action_sampler.proximity_directions(tensor([idx]))
        action = planner.action_sampler.directions_actions(tensor([idx]), directions)[0]
        assert torch.all(directions > 0)  # check direction towards box
        assert torch.all(-params.action_range * time_step <= action)
        assert torch.all(action <= params.action_range * time_step)


def test_continuation_actions() -> None:
    params = ParameterContainer()
    params.parse_params("box_push", "test")
    planner = planner_setup(params, True)
    sorted_ids = planner.graph.sorted_progress_ids(params.reward_based)
    time_step = params.action_time_step
    for _ in range(100):
        idx = sorted_ids[torch.randint(low=0, high=len(sorted_ids), size=()).item()]
        directions = planner.action_sampler.continuation_directions(tensor([idx]))
        action = planner.action_sampler.directions_actions(tensor([idx]), directions)[0]
        start_actions = planner.graph.start_actions[idx]
        end_actions = planner.graph.end_actions[idx]
        if torch.all(start_actions == end_actions):  # check for staying
            assert torch.all(directions == 0)
        else:  # check for continuation
            assert torch.any(directions != 0)
        assert torch.all(-params.action_range * time_step <= action)
        assert torch.all(action <= params.action_range * time_step)


def test_free_gradient_actions() -> None:
    """Checks that pusher moves towards its goal"""
    params = ParameterContainer()
    params.parse_params("box_push", "test")
    params.reward_distance_scaling = torch.diag(torch.tensor([0, 1, 0, 0.0]))
    planner = planner_setup(params, True)
    sorted_ids = planner.graph.sorted_progress_ids(params.reward_based)
    time_step = params.action_time_step
    for _ in range(100):
        idx = sorted_ids[torch.randint(low=0, high=len(sorted_ids), size=()).item()]
        action = planner.action_sampler.gradient_actions(tensor([idx]))[0]
        action_sign = torch.sign(action)
        required_sign = torch.sign(params.goal_state[1] - planner.graph.states[idx, 1])
        assert torch.all(action_sign == required_sign)  # check direction towards goal
        assert torch.all(
            -params.action_range * time_step <= action * 0.99
        )  # scale down slightly for numerics
        assert torch.all(
            action * 0.99 <= params.action_range * time_step
        )  # scale down slightly for numerics


def test_contact_gradient_actions() -> None:
    """Checks that pusher pushes box moves towards the goal"""
    params = ParameterContainer()
    params.parse_params("box_push", "test")
    params.reward_distance_scaling = torch.diag(torch.tensor([1, 0, 0, 0.0]))
    params.goal_state = torch.tensor([3, 0, 0, 0.0])
    params.goal_bound_lower = torch.tensor([3, 0, 0, 0])
    params.goal_bound_upper = torch.tensor([3, 0, 0, 0])
    planner = planner_setup(params, True)
    sorted_ids = planner.graph.sorted_progress_ids(params.reward_based)
    time_step = params.action_time_step

    case_count = torch.zeros(3)
    for _ in range(1000):  # run enough samples to catch all cases
        idx = sorted_ids[torch.randint(low=0, high=len(sorted_ids), size=()).item()]
        action = planner.action_sampler.gradient_actions(tensor([idx]))[0]
        action_sign = torch.sign(action)
        state = planner.graph.states[idx]

        if state[0] - 0.5 - state[1] < 1e-8:  # in contact
            required_sign = (
                1 if state[0] < params.goal_state[0] else -1
            )  # direction towards goal
        elif state[0] - 0.5 - state[1] > 1e-2:  # not in contact
            required_sign = 0  # no gradient available
        else:  # kind of in between -> no check
            required_sign = int(action_sign.item())
        case_count[required_sign + 1] += 1

        assert torch.all(action_sign == required_sign)
        assert torch.all(
            -params.action_range * time_step <= action * 0.99
        )  # scale down slightly for numerics
        assert torch.all(
            action * 0.99 <= params.action_range * time_step
        )  # scale down slightly for numerics
    assert torch.all(case_count != 0)
