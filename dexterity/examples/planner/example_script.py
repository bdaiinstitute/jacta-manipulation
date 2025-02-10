# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
import sys

from dexterity.jacta_planner.dynamics.mujoco_dynamics import MujocoPlant
from dexterity.jacta_planner.planner.action_sampler import ActionSampler
from dexterity.jacta_planner.planner.graph import Graph
from dexterity.jacta_planner.planner.graph_visuals import display_3d_graph
from dexterity.jacta_planner.planner.graph_worker import ExplorerWorker, RelatedGoalWorker, SingleGoalWorker
from dexterity.jacta_planner.planner.logger import Logger
from dexterity.jacta_planner.planner.parameter_container import ParameterContainer
from dexterity.jacta_planner.planner.planner import Planner
from dexterity.jacta_planner.visuals.mujoco_visualizer import MujocoRenderer


def confirm_quit() -> bool:
    while True:
        try:
            inpt = input("Quit? ([y]/n): ").lower()
            if inpt.startswith("n"):
                return False
            elif inpt == "" or inpt.startswith("y"):
                return True
        except EOFError:
            return True


def main(relative_config_dir: str) -> None:
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Join the current directory with the relative path
    config_dir = os.path.join(current_dir, relative_config_dir)

    valid_tasks = [file[:-4] for file in os.listdir(config_dir)]
    valid_planners = ["single_goal", "multi_goal", "learning", "primitives", "exploration"]

    if len(sys.argv) > 2 and sys.argv[1] in valid_tasks and sys.argv[2] in valid_planners:
        test_flag = "test" in locals()  # e.g., to not print info during tests
        task = sys.argv[1]
        planner_example = sys.argv[2]
        params = ParameterContainer(f"dexterity/examples/jacta_planner/config/{task}.yml")
        plant = MujocoPlant(params=params)
        renderer = MujocoRenderer(plant=plant, time_step=plant.sim_time_step)
        graph = Graph(plant, params)
        logger = Logger(graph, params)
        action_sampler = ActionSampler(plant, graph, params)

        def callback(graph: Graph, logger: Logger) -> None:
            display_3d_graph(
                graph, logger, renderer.visualizer, vis_scale=params.vis_scale, vis_indices=params.vis_indices
            )

        match planner_example:
            case "single_goal" | "primitives":
                graph_worker = SingleGoalWorker(plant, graph, action_sampler, logger, params)
            case "multi_goal":
                graph_worker = RelatedGoalWorker(plant, graph, action_sampler, logger, params)
            case "exploration":
                graph_worker = ExplorerWorker(plant, graph, action_sampler, logger, params)
            case _:
                print("Select a valid example from the following: ")
                print(valid_planners)
                return

        if not test_flag and params.callback_period > 0:
            graph_worker.callback = callback
            graph_worker.callback_period = params.callback_period

        planner = Planner(plant, graph, action_sampler, graph_worker, logger, params, verbose=not test_flag)

        params.steps_per_goal = params.steps_per_goal if not test_flag else 10
        planner.search()
        search_index = 0
        state_trajectory = planner.shortest_path_trajectory(search_index=search_index)

        if not test_flag:
            planner.plot_search_results()
            renderer.show(state_trajectory, goal=graph.goal_states[search_index])
            callback(graph, logger)
            confirm_quit()

    else:
        print("Include a valid planner and task as a cli argument. Task options are: ")
        print([task[:-4] for task in os.listdir("config/task/")])
        print("Planner options are: ")
        print(valid_planners)


if __name__ == "__main__":
    main("config/task/")
