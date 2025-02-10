# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import torch

from dexterity.jacta_planner.environments import DexterityEnv
from dexterity.jacta_planner.planner.parameter_container import ParameterContainer
from dexterity.jacta_planner.visuals.mujoco_visualizer import MujocoRenderer

device = torch.device("cuda")

task_name = "planar_hand"  # or "spot_floating_tire"

params = ParameterContainer(f"dexterity/examples/jacta_planner/config/{task_name}.yml")

# TEMP
params.success_progress_threshold = 0.05
params.learner_trajectory_length = 50
params.num_envs = 2

env = DexterityEnv(params)

renderer = MujocoRenderer(plant=env.plant)

env.reset()

for _ in range(params.learner_trajectory_length - 1):
    action = env.uniform_random_action()
    *_, info = env.step(action)

    for state in env.state_trajectory[0].cpu().numpy():
        renderer.render(state[: env.plant.model.nq])

renderer.play(wait_for_input=True)
