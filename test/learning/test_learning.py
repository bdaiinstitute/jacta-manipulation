# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
import torch
from tensordict import TensorDict

from jactamanipulation.learning.learner import Learner
from jactamanipulation.learning.replay_buffer import ReplayBuffer
from jactamanipulation.planner.dynamics.mujoco_dynamics import MujocoPlant
from jactamanipulation.planner.planner.action_sampler import ActionSampler
from jactamanipulation.planner.planner.graph import Graph
from jactamanipulation.planner.planner.graph_worker import ExplorerWorker
from jactamanipulation.planner.planner.logger import Logger
from jactamanipulation.planner.planner.parameter_container import ParameterContainer
from jactamanipulation.planner.planner.planner import Planner
from jactamanipulation.planner.planner.types import ActionType, set_default_device_and_dtype


def learner_setup(search: bool, learn: bool) -> Learner:
    set_default_device_and_dtype()
    params = ParameterContainer("dexterity/examples/learning/config/box_push.yml")
    params.action_time_step = 0.4
    params.action_steps_max = 1
    params.action_types = [ActionType.RANGED]
    params.action_distribution = torch.tensor([1.0])

    plant = MujocoPlant(params=params)
    graph = Graph(plant, params)
    graph.set_start_states(params.start_state.unsqueeze(0))
    logger = Logger(graph, params)
    action_sampler = ActionSampler(plant, graph, params)
    graph_worker = ExplorerWorker(plant, graph, action_sampler, logger, params)

    planner = Planner(plant, graph, action_sampler, graph_worker, logger, params, verbose=False)

    replay_buffer = ReplayBuffer(params)
    learner = Learner(plant, graph, replay_buffer, params, verbose=False)

    if search:
        planner.search()

    if learn:
        learner.params.learner_cycles = 2
        learner.params.learner_rollouts = 10
        learner.params.learner_batches = 2
        learner.params.learner_early_stop = 1.0
        learner.train(num_epochs=2)

    return learner


def test_reward_function() -> None:
    learner = learner_setup(False, False)
    params = learner.params
    replay_buffer = learner.replay_buffer
    learner.final_success_distance = 0.2

    start_states = params.start_state.unsqueeze(0)
    # Add initial state
    initial_state = TensorDict(
        {
            "states": start_states,
            "actions": torch.zeros(1, 7),
            "rewards": torch.tensor([[0.0]]),
            "done": torch.tensor([[False]]),
            "next_states": params.start_state.unsqueeze(0),
            "goals": params.goal_state.unsqueeze(0),
        },
        batch_size=1,
    )
    replay_buffer.extend(initial_state)

    # Add two new states
    state_difference = params.goal_state - params.start_state
    close_state = params.goal_state - state_difference * 0.19
    far_state = params.goal_state - state_difference * 0.21

    new_states = TensorDict(
        {
            "states": torch.stack([close_state, far_state]),
            "actions": torch.zeros(2, 7),
            "rewards": torch.zeros(2, 1),
            "done": torch.tensor([[False], [False]]),
            "next_states": torch.stack([params.goal_state, params.goal_state]),
            "goals": torch.stack([params.goal_state, params.goal_state]),
        },
        batch_size=2,
    )
    replay_buffer.extend(new_states)

    # Sample the two new states
    sample = replay_buffer[1:3]

    # Check rewards for the two new states
    rewards, _ = learner.reward_function(
        start_states=start_states,
        current_states=sample["states"],
        goal_states=sample["goals"],
    )
    assert torch.all(rewards == torch.tensor([0, -1])), f"Expected rewards [0, -1], got {rewards}"


def test_graph_rollout() -> None:
    learner = learner_setup(True, False)
    replay_buffer = learner.replay_buffer

    # Perform a graph rollout
    rollout_data = learner.graph_rollout()

    # Check if the rollout data has the correct structure
    assert isinstance(rollout_data, TensorDict), "Rollout data should be a TensorDict"
    assert set(rollout_data.keys()) == {"states", "relative_actions", "start_states", "goal_states", "next_states"}

    # Check that the rollout contains non-zero values
    assert torch.any(rollout_data["states"] != 0.0)
    assert torch.any(rollout_data["relative_actions"] != 0.0)
    assert torch.any(rollout_data["next_states"] != 0.0)
    assert torch.any(rollout_data["goal_states"] != 0.0)

    # Check that states are consistent throughout the trajectory
    assert torch.all(rollout_data["start_states"] == rollout_data["start_states"][0])
    assert torch.all(rollout_data["goal_states"] == rollout_data["goal_states"][0])
    assert torch.all(rollout_data["next_states"][:-1] == rollout_data["states"][1:])

    # Add the rollout data to the replay buffer
    replay_buffer.extend(rollout_data)

    # Sample the entire trajectory from the buffer
    path_length = rollout_data.shape[0]
    sampled_trajectory = replay_buffer[-path_length:]

    # Check if the sampled trajectory matches the original rollout data
    for key in rollout_data.keys():
        assert torch.all(
            rollout_data[key] == sampled_trajectory[key]
        ), f"Mismatch in {key} between original and sampled trajectory"


def test_policy_rollout() -> None:
    learner = learner_setup(False, False)
    params = learner.params
    replay_buffer = learner.replay_buffer
    a_min = params.action_bound_lower
    a_max = params.action_bound_upper
    a_range = params.action_range
    time_step = params.action_time_step

    # Perform a policy rollout
    rollout_data = learner.policy_rollout()

    # Check if the rollout data has the correct structure
    assert isinstance(rollout_data, TensorDict), "Rollout data should be a TensorDict"
    expected_keys = {"states", "relative_actions", "start_states", "goal_states", "next_states"}
    assert set(rollout_data.keys()) == expected_keys

    # Check the shape of the rollout data
    assert (
        rollout_data.batch_size[0] == params.learner_trajectory_length - 1
    ), f"Expected trajectory length {params.learner_trajectory_length - 1}, got {rollout_data.batch_size[0]}"

    # Check that the rollout contains non-zero values
    assert torch.any(rollout_data["states"] != 0.0), "All states are zero"
    assert torch.any(rollout_data["relative_actions"] != 0.0), "All relative_actions are zero"
    assert torch.any(rollout_data["next_states"] != 0.0), "All next_states are zero"

    # Check start, goal, state vs next states
    assert torch.all(rollout_data["start_states"] == rollout_data["start_states"][0]), "Start states are not consistent"
    assert torch.all(rollout_data["goal_states"] == rollout_data["goal_states"][0]), "Goal states are not consistent"
    assert torch.all(rollout_data["next_states"][:-1] == rollout_data["states"][1:])

    # Check action bounds
    assert torch.all(a_min - 0.5 * a_range * time_step <= rollout_data["relative_actions"]), "Actions below lower bound"
    assert torch.all(rollout_data["relative_actions"] <= a_max + 0.5 * a_range * time_step), "Actions above upper bound"

    # Add the rollout data to the replay buffer
    replay_buffer.extend(rollout_data)

    # Sample the entire trajectory from the buffer
    sampled_trajectory = replay_buffer[-rollout_data.batch_size[0] :]

    # Check if the sampled trajectory matches the original rollout data
    for key in rollout_data.keys():
        assert torch.all(
            rollout_data[key] == sampled_trajectory[key]
        ), f"Mismatch in {key} between original and sampled trajectory"


def test_actor_actions() -> None:
    learner = learner_setup(False, False)
    params = learner.params
    time_step = params.action_time_step

    learner.actor.train()
    states = params.start_state.unsqueeze(0)
    goal_states = params.goal_state.unsqueeze(0)
    action1 = learner.actor_actions(states, goal_states)
    action2 = learner.actor_actions(states, goal_states)
    assert all(action1 != action2)
    assert any(action1 != 0)
    assert torch.all(-params.action_range * time_step <= action1)
    assert torch.all(-params.action_range * time_step <= action2)
    assert torch.all(action1 <= params.action_range * time_step)
    assert torch.all(action2 <= params.action_range * time_step)

    learner.actor.eval()
    action1 = learner.actor_actions(states, goal_states)
    action2 = learner.actor_actions(states, goal_states)
    assert all(action1 == action2)
    assert any(action1 != 0)
    assert torch.all(-params.action_range * time_step <= action1)
    assert torch.all(action1 <= params.action_range * time_step)


def test_sampling() -> None:
    learner = learner_setup(True, True)
    params = learner.params
    replay_buffer = learner.replay_buffer

    rollout_data = learner.graph_rollout()
    replay_buffer.extend(rollout_data)

    expected_keys = {"states", "relative_actions", "next_states", "start_states", "goal_states", "index"}
    # Perform sampling twice
    for _ in range(2):
        sample = replay_buffer.sample()
        assert isinstance(sample, TensorDict), "Sampled data should be a TensorDict"
        assert set(sample.keys()) == expected_keys

        # Check shapes
        batch_size = params.learner_batch_size
        assert sample["states"].shape[0] == batch_size, "Incorrect batch size for states"
        assert sample["relative_actions"].shape[0] == batch_size, "Incorrect batch size for actions"
        assert sample["next_states"].shape[0] == batch_size, "Incorrect batch size for next_states"
        assert sample["start_states"].shape[0] == batch_size, "Incorrect batch size for start_states"
        assert sample["goal_states"].shape[0] == batch_size, "Incorrect batch size for goal_states"

        # Check that sampled data is within the replay buffer
        buffer_data = replay_buffer[:]
        for key in sample.keys():
            assert torch.all(torch.isin(sample[key], buffer_data[key]))


def test_eval_agent() -> None:
    learner = learner_setup(False, False)

    # run twice to see if reset works
    learner.eval_agent()
    final_success, _ = learner.eval_agent()

    # check results
    assert torch.all(final_success >= 0.0)
    assert torch.all(final_success <= 1.0)
