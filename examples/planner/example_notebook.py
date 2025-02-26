# %%
# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
from jacta.planner.core.action_sampler import ActionSampler
from jacta.planner.core.graph import Graph
from jacta.planner.core.graph_visuals import display_3d_graph
from jacta.planner.core.graph_worker import (
    ExplorerWorker,
    RelatedGoalWorker,
    SingleGoalWorker,
)
from jacta.planner.core.logger import Logger
from jacta.planner.core.parameter_container import ParameterContainer
from jacta.planner.core.planner import Planner
from jacta.planner.dynamics.mujoco_dynamics import MujocoPlant
from jacta.visualizers.meshcat.visuals import TrajectoryVisualizer

# %%
task = "planar_hand"  # Set desired example here
planner_example = "single_goal"

params = ParameterContainer()
params.parse_params(task, planner_example)

# %%
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
match planner_example:
    case "single_goal":
        graph_worker = SingleGoalWorker(plant, graph, action_sampler, logger, params)
    case "multi_goal":
        graph_worker = RelatedGoalWorker(plant, graph, action_sampler, logger, params)
    case "exploration":
        graph_worker = ExplorerWorker(plant, graph, action_sampler, logger, params)
    case _:
        print("Select a valid example from the following: ")
        print(["single_goal", "multi_goal", "learning", "exploration"])

if params.callback_period > 0:
    graph_worker.callback = callback
    graph_worker.callback_period = params.callback_period

planner = Planner(plant, graph, action_sampler, graph_worker, logger, params)

# %%
planner.search()

# %%
search_index = 0
state_trajectory = planner.shortest_path_trajectory(search_index=search_index)

planner.plot_search_results()
visualizer.show(state_trajectory, goal_state=graph.goal_states[search_index])
callback(graph, logger)
