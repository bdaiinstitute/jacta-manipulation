# %%
# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from dexterity.jacta_planner.dynamics.mujoco_dynamics import MujocoPlant
from dexterity.jacta_planner.planner.action_sampler import ActionSampler
from dexterity.jacta_planner.planner.graph import Graph
from dexterity.jacta_planner.planner.graph_visuals import display_3d_graph
from dexterity.jacta_planner.planner.graph_worker import (
    ExplorerWorker,
    RelatedGoalWorker,
    SingleGoalWorker,
)
from dexterity.jacta_planner.planner.logger import Logger
from dexterity.jacta_planner.planner.parameter_container import ParameterContainer
from dexterity.jacta_planner.planner.planner import Planner
from dexterity.jacta_planner.visuals.mujoco_visualizer import MujocoRenderer

# %%
task = "planar_hand"  # Set desired example here
planner_example = "single_goal"

params = ParameterContainer(f"dexterity/examples/jacta_planner/config/{task}.yml")

# %%
plant = MujocoPlant(params=params)
renderer = MujocoRenderer(plant=plant, time_step=plant.sim_time_step)


def callback(graph: Graph, logger: Logger) -> None:
    display_3d_graph(graph, logger, renderer.visualizer, vis_scale=params.vis_scale, vis_indices=params.vis_indices)


graph = Graph(plant, params)
logger = Logger(graph, params)
action_sampler = ActionSampler(plant, graph, params)
match planner_example:
    case "single_goal" | "primitives":
        graph_worker = SingleGoalWorker(plant, graph, action_sampler, logger, params)
    case "multi_goal":
        graph_worker = RelatedGoalWorker(plant, graph, action_sampler, logger, params)
    case "exploration":
        graph_worker = ExplorerWorker(plant, graph, action_sampler, logger, params)
    case _:
        print("Select a valid example from the following: ")
        print(["single_goal", "multi_goal", "learning", "primitives", "exploration"])

if params.callback_period > 0:
    graph_worker.callback = callback
    graph_worker.callback_period = params.callback_period

planner = Planner(plant, graph, action_sampler, graph_worker, logger, params, verbose=True)

# %%
planner.search()

# %%
search_index = 0
state_trajectory = planner.shortest_path_trajectory(search_index=search_index)

planner.plot_search_results()
renderer.show(state_trajectory, goal=graph.goal_states[search_index])
callback(graph, logger)

# %%
while True:
    pass
