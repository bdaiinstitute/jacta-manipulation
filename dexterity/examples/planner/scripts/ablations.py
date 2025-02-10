# %%
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import torch

from dexterity.jacta_planner.dynamics.mujoco_dynamics import MujocoPlant
from dexterity.jacta_planner.planner.action_sampler import ActionSampler
from dexterity.jacta_planner.planner.graph import Graph, sample_random_goal_states
from dexterity.jacta_planner.planner.graph_worker import ExplorerWorker
from dexterity.jacta_planner.planner.logger import Logger
from dexterity.jacta_planner.planner.parameter_container import ParameterContainer
from dexterity.jacta_planner.planner.planner import Planner
from dexterity.verification.visuals import TrajectoryVisualizer

for task in [
    "box_push",
    "box_push_2d",
    "planar_hand",
    "floating_hand",
    "allegro_hand",
    "spot_standing_box_lift",
    "spot_lying_ball",
    "spot_bimanual_stool_lift",
]:

    print(task)

    params = ParameterContainer(f"dexterity/examples/jacta_learning/config/{task}.yml")

    plant = MujocoPlant(params=params)
    visualizer = TrajectoryVisualizer(params=params, sim_time_step=plant.sim_time_step)

    graph = Graph(plant, params)
    logger = Logger(graph, params)
    action_sampler = ActionSampler(plant, graph, params)
    graph_worker = ExplorerWorker(plant, graph, action_sampler, logger, params)
    planner = Planner(plant, graph, action_sampler, graph_worker, logger, params, verbose=True)

    print(str(params))

    data_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    len_data_range = len(data_range)

    success_avg = torch.zeros(len_data_range)
    rel_dist_avg = torch.zeros(len_data_range)
    len_avg = torch.zeros(len_data_range)

    for val_ind, val in enumerate(data_range):
        values = {"action_steps_max": val}
        params.update(values)
        print("action_steps_max:")
        print(params.action_steps_max)

        for i in range(5):
            params.seed = i
            planner.reset()

            graph.set_start_states(params.start_state.unsqueeze(0))
            planner.search()
            graph.activate_all_nodes()

            for _ in range(10):
                graph.change_sub_goal_states(sample_random_goal_states(plant, params))

                best_ids = graph.get_best_id(reward_based=False)
                best_distances = graph.scaled_goal_distances[best_ids]
                root_ids = graph.root_ids[best_ids]
                root_distances = graph.scaled_goal_distances[root_ids]
                relative_distances = best_distances / root_distances

                success_avg[val_ind] += (relative_distances < 0.5).item() / 50
                rel_dist_avg[val_ind] += relative_distances.item() / 50
                len_avg[val_ind] += len(planner.shortest_path_data()[0]) / 50

    print("demonstration eval (succ., rel. dist., length)")
    print(", ".join(map(str, [success_avg[ind].item() for ind in range(len_data_range)])))
    print(", ".join(map(str, [rel_dist_avg[ind].item() for ind in range(len_data_range)])))
    print(", ".join(map(str, [len_avg[ind].item() for ind in range(len_data_range)])))
