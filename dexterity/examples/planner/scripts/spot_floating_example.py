# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import torch
from tqdm import trange

from dexterity.jacta_planner.dynamics.mujoco_dynamics import MujocoPlant
from dexterity.jacta_planner.planner.action_processor import SpotFloatingActionProcessor
from dexterity.jacta_planner.planner.parameter_container import ParameterContainer
from dexterity.jacta_planner.planner.types import ActionMode
from dexterity.jacta_planner.visuals.mujoco_visualizer import MujocoRenderer

task_name = "spot_floating_tire"

params = ParameterContainer(f"dexterity/examples/jacta_planner/config/{task_name}.yml")
params.action_start_mode = ActionMode.RELATIVE_TO_CURRENT_STATE

plant = MujocoPlant(params=params)

action_processor = SpotFloatingActionProcessor(params, plant.actuated_pos)

states = params.start_state.unsqueeze(0)

renderer = MujocoRenderer(plant=plant)

render_state = states.cpu().numpy()[:, : plant.model.nq]
renderer.render(render_state)

for t in trange(50):
    delta_base = [0.1, 0.0, 0.0]
    if t < 10:
        delta_base = [0.0, 0.0, +0.1]
    delta_arm = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    relative_actions = torch.tensor([delta_base + delta_arm])
    _, _, action_trajectories = action_processor(
        relative_actions=relative_actions,
        current_states=states,
    )
    states, _, states_trajectory = plant.dynamics(
        states,
        action_trajectories,
    )
    for states_t in states_trajectory[0]:
        render_state = states_t.cpu().numpy()[: plant.model.nq]
        renderer.render(render_state)

renderer.play(wait_for_input=True)
