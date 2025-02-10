# %%
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
import cProfile
import os

from bdai_core.os.path import get_bdai_path

from dexterity.jacta_planner.benchmarking.profiling import run_example

# %%
examples_list = [
    ["planner", "allegro_hand_eigenspace"],
    ["planner", "allegro_hand"],
    ["planner", "bimanual_kuka_allegro"],
    ["planner", "bimanual_station"],
    ["planner", "box_push"],
    ["planner", "cartpole"],
    ["planner", "floating_hand"],
    ["planner", "kuka_allegro"],
    ["planner", "planar_hand_eigenspace"],
    ["planner", "planar_hand"],
    ["planner", "satellite_offcenter"],
    ["planner", "satellite"],
    ["planner", "spot_bimanual_box"],
    ["planner", "spot_floating_box"],
    ["planner", "spot_standing_box"],
    ["rrt", "allegro_hand"],
    ["rrt", "bimanual_station"],
    ["rrt", "floating_hand"],
    ["rrt", "planar_hand"],
]

# %%
for example in examples_list:
    profiler = cProfile.Profile()

    profiler.enable()
    example_type = example[0]
    example_name = example[1]
    run_example(example_type, example_name, test_mode=True)
    profiler.disable()

    profiling_folder = get_bdai_path() + "/projects/dexterity/log/profiling"
    os.makedirs(profiling_folder, exist_ok=True)
    profiler.dump_stats(profiling_folder + f"/{example[0]}_{example[1]}.prof")


# %%
