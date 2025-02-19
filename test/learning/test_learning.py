# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
import torch
from jacta.planner.dynamics.mujoco_dynamics import MujocoPlant
from jacta.learning.learner import Learner
from jacta.learning.replay_buffer import ReplayBuffer
from jacta.planner.planner.action_sampler import ActionSampler
from jacta.planner.planner.graph import Graph
from jacta.planner.planner.graph_worker import ExplorerWorker
from jacta.planner.planner.logger import Logger
from jacta.planner.planner.parameter_container import ParameterContainer
from jacta.planner.planner.planner import Planner
from jacta.planner.planner.types import ActionType, set_default_device_and_dtype
from torch import tensor


def learner_setup(search: bool, learn: bool, use_planner_exploration: bool = False) -> Learner:
    set_default_device_and_dtype()
    params = ParameterContainer()
    params.parse_params("box_push", "test")
    params.action_time_step = 0.4
    params.action_steps_max = 1
    params.action_types = [ActionType.RANGED]
    params.action_distribution = torch.tensor([1.0])
    params.learner_use_planner_exploration = use_planner_exploration

    plant = MujocoPlant(params)
    graph = Graph(plant, params)
    graph.set_start_states(params.start_state.unsqueeze(0))
    logger = Logger(graph, params)
    action_sampler = ActionSampler(plant, graph, params)
    graph_worker = ExplorerWorker(plant, graph, action_sampler, logger, params)

    planner = Planner(plant, graph, action_sampler, graph_worker, logger, params, verbose=False)

    replay_buffer = ReplayBuffer(plant, params)
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
    graph = learner.graph
    replay_buffer = learner.replay_buffer
    learner.current_success_distance = 0.2

    # Add initial state
    root_id = tensor([0], dtype=torch.int64)
    state = graph.states[root_id]
    start_action = graph.start_actions[root_id]
    end_action = graph.end_actions[root_id]
    relative_action = graph.relative_actions[root_id]
    parent_id = tensor([replay_buffer.next_temporary_id], dtype=torch.int64)
    learner.replay_buffer.add_nodes(
        root_id,
        parent_id,
        state,
        start_action,
        end_action,
        relative_action,
        temporary=True,
        sub_goal_state=graph.sub_goal_states[0],
    )

    # Add two new states
    root_ids = torch.tile(root_id, (2,))
    parent_ids = torch.tile(parent_id, (2,))
    state_difference = params.goal_state - params.start_state
    states = params.goal_state - torch.stack((state_difference * 0.19, state_difference * 0.21))
    actions = tensor([[0.0], [0.0]])
    relative_actions = tensor([[0.0], [0.0]])
    action_time_step = 1.0
    ids = learner.replay_buffer.add_nodes(
        root_ids,
        parent_ids,
        states,
        actions,
        actions,
        relative_actions,
        action_time_step,
        sub_goal_state=graph.sub_goal_states[0],
    )

    # Check rewards for the two new states
    rewards, _ = learner.reward_function(
        learner.replay_buffer, ids, torch.stack((params.goal_state, params.goal_state))
    )
    assert all(rewards == tensor([0, -1]))


def test_graph_rollout() -> None:
    learner = learner_setup(True, False)
    params = learner.params
    graph = learner.graph
    replay_buffer = learner.replay_buffer

    learner_ids = learner.graph_rollout()
    best_id = graph.get_best_id(reward_based=False)
    path_to_goal = graph.shortest_path_to(best_id)[-params.learner_trajectory_length :]
    padding_ids = torch.ones(params.learner_trajectory_length - len(path_to_goal), dtype=int) * path_to_goal[0]
    planner_ids = torch.concatenate((padding_ids, path_to_goal))
    assert torch.all(replay_buffer.states[learner_ids] == graph.states[planner_ids])
    assert torch.all(replay_buffer.start_actions[learner_ids] == graph.start_actions[planner_ids])
    assert torch.all(replay_buffer.end_actions[learner_ids] == graph.end_actions[planner_ids])
    assert torch.all(replay_buffer.relative_actions[learner_ids] == graph.relative_actions[planner_ids])
    assert torch.any(replay_buffer.states[learner_ids] != 0.0)
    assert torch.any(replay_buffer.start_actions[learner_ids] != 0.0)
    assert torch.any(replay_buffer.end_actions[learner_ids] != 0.0)
    assert torch.any(replay_buffer.relative_actions[learner_ids] != 0.0)
    assert torch.any(replay_buffer.learning_goals[learner_ids] != 0.0)
    assert torch.all(replay_buffer.root_ids[learner_ids] == replay_buffer.first_learner_id)


def test_policy_rollout() -> None:
    learner = learner_setup(False, False)
    params = learner.params
    replay_buffer = learner.replay_buffer
    a_min = params.action_bound_lower
    a_max = params.action_bound_upper
    a_range = params.action_range
    time_step = params.action_time_step

    node_ids, _ = learner.policy_rollout()
    assert all(node_ids[:-1] + 1 == node_ids[1:])
    assert replay_buffer.parents[node_ids[0]] == replay_buffer.parents[node_ids[1]]
    assert all(
        node_ids
        == torch.arange(
            replay_buffer.first_learner_id, replay_buffer.first_learner_id + params.learner_trajectory_length
        )
    )
    assert torch.any(replay_buffer.states[node_ids] != 0.0)
    assert torch.any(replay_buffer.start_actions[node_ids] != 0.0)
    assert torch.any(replay_buffer.end_actions[node_ids] != 0.0)
    assert torch.any(replay_buffer.relative_actions[node_ids] != 0.0)
    assert torch.all(a_min - 0.5 * a_range * time_step <= replay_buffer.start_actions[node_ids])
    assert torch.all(replay_buffer.start_actions[node_ids] <= a_max + 0.5 * a_range * time_step)
    assert torch.all(a_min - 0.5 * a_range * time_step <= replay_buffer.end_actions[node_ids])
    assert torch.all(replay_buffer.end_actions[node_ids] <= a_max + 0.5 * a_range * time_step)


def test_actor_actions() -> None:
    learner = learner_setup(False, False)
    params = learner.params
    time_step = params.action_time_step

    learner.actor.train()
    action1 = learner.actor_actions(learner.actor, [0, 0], time_step)
    action2 = learner.actor_actions(learner.actor, [0, 0], time_step)
    assert all(action1 != action2)
    assert any(action1 != 0)
    assert torch.all(-params.action_range * time_step <= action1)
    assert torch.all(-params.action_range * time_step <= action2)
    assert torch.all(action1 <= params.action_range * time_step)
    assert torch.all(action2 <= params.action_range * time_step)

    learner.actor.eval()
    action1 = learner.actor_actions(learner.actor, [0, 0], time_step)
    action2 = learner.actor_actions(learner.actor, [0, 0], time_step)
    assert all(action1 == action2)
    assert any(action1 != 0)
    assert torch.all(-params.action_range * time_step <= action1)
    assert torch.all(action1 <= params.action_range * time_step)


def test_sampling() -> None:
    learner = learner_setup(True, True)
    params = learner.params
    replay_buffer = learner.replay_buffer
    action_normalization_scaling = params.action_range * params.action_time_step

    states, actions, rewards, next_states, goals, current_ids, next_ids, her_ids = replay_buffer.sampling(
        50, 0, learner.reward_function
    )
    assert torch.all(replay_buffer.states[current_ids] == states)
    # actions from current node are stored in next node
    assert torch.all(replay_buffer.relative_actions[next_ids] / action_normalization_scaling == actions)
    assert torch.all(replay_buffer.states[next_ids] == next_states)
    assert torch.all(replay_buffer.learning_goals[current_ids] == goals)
    assert torch.all((current_ids == next_ids) + (current_ids == replay_buffer.parents[next_ids]))

    states, actions, rewards, next_states, goals, current_ids, next_ids, her_ids = replay_buffer.sampling(
        50, 1, learner.reward_function
    )
    reward_indices = current_ids == her_ids
    assert torch.all(replay_buffer.states[current_ids] == states)
    # actions from current node are stored in next node
    assert torch.all(replay_buffer.relative_actions[next_ids] / action_normalization_scaling == actions)
    assert torch.all(replay_buffer.states[next_ids] == next_states)
    assert torch.all(replay_buffer.states[her_ids] == goals)
    assert torch.all((current_ids == next_ids) + (current_ids == replay_buffer.parents[next_ids]))
    assert torch.all(rewards[reward_indices] == 0)
    assert torch.all(current_ids <= her_ids)


def test_eval_agent() -> None:
    learner = learner_setup(False, False)
    params = learner.params
    replay_buffer = learner.replay_buffer

    # run twice to see if reset works
    learner.eval_agent()
    final_success, current_success, relative_distance = learner.eval_agent()

    # check if only temporary buffer is filled
    assert torch.any(replay_buffer.states[0 : replay_buffer.first_learner_id] != 0)
    assert torch.all(replay_buffer.states[replay_buffer.first_learner_id :] == 0)

    # check that full trajectories are stored
    last_ids = torch.zeros(params.learner_evals, dtype=int)
    i = 0
    for index in range(params.learner_trajectory_length * params.learner_evals):
        if index % params.learner_trajectory_length == 0:
            assert replay_buffer.parents[index] == index
        else:
            assert replay_buffer.parents[index] == index - 1
            if (index + 1) % params.learner_trajectory_length == 0:
                last_ids[i] = index
                i += 1

    # check results
    relative_distances = learner.relative_distances_to(replay_buffer, last_ids, replay_buffer.learning_goals[last_ids])
    final_success_direct = torch.sum(relative_distances <= learner.final_success_distance) / params.learner_evals
    current_success_direct = torch.sum(relative_distances <= learner.current_success_distance) / params.learner_evals
    relative_distance_direct = torch.sum(relative_distances) / params.learner_evals
    assert final_success == final_success_direct
    assert current_success == current_success_direct
    assert relative_distance == relative_distance_direct
    assert len(torch.unique(relative_distances)) == params.learner_evals


def test_graph_buffer() -> None:
    learner = learner_setup(True, False)
    params = learner.params
    replay_buffer = learner.replay_buffer

    # run three times to get past end of buffer
    node_ids, _ = learner.policy_rollout()
    start_id = replay_buffer.first_learner_id
    end_id = start_id + params.learner_trajectory_length
    assert torch.all(node_ids == torch.arange(start_id, end_id))

    node_ids = learner.graph_rollout()
    start_id = end_id
    end_id = start_id + params.learner_trajectory_length
    assert torch.all(node_ids == torch.arange(start_id, end_id))

    node_ids, _ = learner.policy_rollout()
    start_id = replay_buffer.first_learner_id
    end_id = start_id + params.learner_trajectory_length
    assert torch.all(node_ids == torch.arange(start_id, end_id))

    # check that only learning buffer is filled
    assert torch.all(replay_buffer.states[0 : replay_buffer.first_learner_id] == 0)
    assert torch.any(replay_buffer.states[replay_buffer.first_learner_id :] != 0)

    # check next and max ids
    assert replay_buffer.next_learner_id == replay_buffer.first_learner_id + params.learner_trajectory_length
    assert replay_buffer.max_learner_id == replay_buffer.states.shape[0] - 1

    # check that full trajectories are stored
    for index in range(params.learner_trajectory_length * params.learner_max_trajectories):
        id = replay_buffer.first_learner_id + index
        if index % params.learner_trajectory_length == 0:
            assert replay_buffer.parents[id] == id
        else:
            assert replay_buffer.parents[id] == id - 1


def test_planner_exploration() -> None:
    learner = learner_setup(False, False, True)
    learner.actor.train()
    initial_state = learner.action_sampler.graph.states[0].clone()
    assert learner.action_sampler.graph.get_active_main_ids() == torch.tensor(0).unsqueeze(0)

    learner.actor.eps = 1.0  # force off policy exploration
    learner.policy_rollout()
    new_state = learner.action_sampler.graph.states[0].clone()
    assert learner.action_sampler.graph.get_active_main_ids() == torch.tensor(0).unsqueeze(0)
    assert torch.any(initial_state != new_state)

    for _ in range(50):
        actions = learner.exploration_function(initial_state.unsqueeze(0))
        assert torch.any(actions != 0.0)
        assert torch.all(actions <= 1.0)
        assert torch.all(actions >= -1.0)
