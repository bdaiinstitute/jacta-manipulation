# %%
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import torch
from benedict import benedict
from jacta.planner.dynamics.mujoco_dynamics import MujocoPlant
from jacta.planner.learning.learner import Learner
from jacta.planner.learning.replay_buffer import ReplayBuffer
from jacta.planner.planner.action_sampler import ActionSampler
from jacta.planner.planner.graph import Graph, sample_random_goal_states
from jacta.planner.planner.graph_visuals import display_3d_graph
from jacta.planner.planner.graph_worker import ExplorerWorker, RolloutWorker
from jacta.planner.planner.logger import Logger
from jacta.planner.planner.parameter_container import ParameterContainer
from jacta.planner.planner.planner import Planner
from jacta.planner.planner.types import ActionType as AT
from jacta.planner.verification.visuals import TrajectoryVisualizer
from matplotlib.pyplot import figure, legend, plot

# %%
task = "planar_hand"  # Set desired example here

params = ParameterContainer()
params.parse_params(task, "learning")

plant = MujocoPlant(params)
visualizer = TrajectoryVisualizer(params=params, sim_time_step=plant.sim_time_step)


def callback(graph: Graph, logger: Logger) -> None:
    display_3d_graph(graph, logger, visualizer.meshcat, vis_scale=params.vis_scale, vis_indices=params.vis_indices)


graph = Graph(plant, params)
graph.set_start_states(params.start_state.unsqueeze(0))
logger = Logger(graph, params)
action_sampler = ActionSampler(plant, graph, params)
graph_worker = ExplorerWorker(plant, graph, action_sampler, logger, params, callback=callback, callback_period=5)
planner = Planner(plant, graph, action_sampler, graph_worker, logger, params, verbose=True)

# %%
planner.search()
graph.activate_all_nodes()

# %%
replay_buffer = ReplayBuffer(plant, params)
learner = Learner(plant, graph, replay_buffer, params)

# %%
learner.pretrain(save_trajectories=True)

# %%
if task == "box_push":
    states = torch.tensor([[8, i, 0, 0] for i in torch.linspace(1, 10, 50)])
    goals = params.goal_state.repeat(len(states), 1)
    obs = learner.state_norm.wrap_obs(states, goals)

    with torch.no_grad():
        actions = learner.actor_expert.target(obs).cpu()

    # policy plot
    plot(states[:, 1], actions[:, 0])

    actions_left = torch.ones((50, 1), dtype=torch.float32, device=learner.critic.device) * -0.5
    actions_zero = torch.ones((50, 1), dtype=torch.float32, device=learner.critic.device) * 0
    actions_right = torch.ones((50, 1), dtype=torch.float32, device=learner.critic.device) * 0.5
    with torch.no_grad():
        values_left = learner.critic(obs, actions_left).cpu()
        values_zero = learner.critic(obs, actions_zero).cpu()
        values_right = learner.critic(obs, actions_right).cpu()

    # q function plot
    figure()
    plot(states[:, 1], values_left, label="left")
    plot(states[:, 1], values_zero, label="zero")
    plot(states[:, 1], values_right, label="right")
    legend(loc="upper left")

# %%
# Rollout actor network on planner
values = {
    "action": benedict(
        {
            "types": [AT.EXPERT],
            "distribution": torch.ones(1),
            "experts": ["network.NetworkSampler"],
            "expert_kwargs": ['{"model_name": "actor_expert.pt"}'],
            "expert_distribution": torch.ones(1),
        }
    ),
    "action_steps_max": 1,
    "steps_per_goal": "self.learner_trajectory_length * 2",
    "num_sub_goals": 0,
}
params.update(values)

graph.reset()
logger.reset()
action_sampler.reset()
graph_worker = RolloutWorker(plant, graph, action_sampler, logger, params)
planner = Planner(plant, graph, action_sampler, graph_worker, logger, params, verbose=True)

graph.change_sub_goal_states(sample_random_goal_states(plant, params)[0])
planner.search()

state_trajectory = planner.path_trajectory(planner.path_data(0, graph.next_main_node_id - 1))
# state_trajectory = planner.shortest_path_trajectory()

planner.plot_search_results()
visualizer.show(state_trajectory, goal_state=graph.sub_goal_states[0])
callback(graph, logger)
# %%
