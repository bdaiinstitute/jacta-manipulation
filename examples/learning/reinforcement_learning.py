# %%
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from jacta.learning.learner import Learner
from jacta.learning.replay_buffer import ReplayBuffer
from jacta.planner.core.action_sampler import ActionSampler
from jacta.planner.core.graph import Graph
from jacta.planner.core.graph_visuals import display_3d_graph
from jacta.planner.core.graph_worker import ExplorerWorker
from jacta.planner.core.logger import Logger
from jacta.planner.core.parameter_container import ParameterContainer
from jacta.planner.core.planner import Planner
from jacta.planner.dynamics.mujoco_dynamics import MujocoPlant
from jacta.visualizers.meshcat.visuals import TrajectoryVisualizer

# %%
task = "allegro_hand"  # Set desired example here

params = ParameterContainer()
params.parse_params(task, "learning")

plant = MujocoPlant(params)
visualizer = TrajectoryVisualizer(params=params, sim_time_step=plant.sim_time_step)


def callback(graph: Graph, logger: Logger) -> None:
    display_3d_graph(
        graph,
        logger,
        visualizer.meshcat,
        vis_scale=params.vis_scale,
        vis_indices=params.vis_indices,
    )


graph = Graph(plant, params)
logger = Logger(graph, params)
action_sampler = ActionSampler(plant, graph, params)
graph_worker = ExplorerWorker(
    plant, graph, action_sampler, logger, params, callback=callback, callback_period=5
)
planner = Planner(
    plant, graph, action_sampler, graph_worker, logger, params, verbose=True
)

# %%
replay_buffer = ReplayBuffer(plant, params)
learner = Learner(plant, graph, replay_buffer, params)

# %%
print(str(params))

# for i in range(5):
# params.seed = i
planner.reset()
learner.reset()

graph.set_start_states(params.start_state.unsqueeze(0))
planner.search()
graph.activate_all_nodes()

# learner.train(num_epochs=300)

# # %%
# values = {
#     "action_types": [AT.EXPERT],
#     "action_distribution": torch.ones(1),
#     "action_experts": ["network.NetworkSampler"],
#     "action_expert_kwargs": ["{}"],
#     "action_expert_distribution": torch.ones(1),
#     "action_steps_max": 1,
#     "steps_per_goal": "self.learner_trajectory_length",
#     "num_sub_goals": 0,
# }
# params.update(values)

# graph.reset()
# logger.reset()
# action_sampler.reset()
# graph_worker = RolloutWorker(plant, graph, action_sampler, logger, params)
# planner = Planner(plant, graph, action_sampler, graph_worker, logger, params, verbose=True)

# graph.change_sub_goal_states(sample_random_goal_states(plant, params))
# planner.search()

state_trajectory = planner.path_trajectory(
    planner.path_data(0, graph.next_main_node_id - 1)
)
# state_trajectory = planner.shortest_path_trajectory()

visualizer.show(state_trajectory, goal_state=graph.sub_goal_states[0])
callback(graph, logger)

while True:
    pass
