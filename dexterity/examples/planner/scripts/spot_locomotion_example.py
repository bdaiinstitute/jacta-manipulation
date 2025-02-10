# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import torch
from tqdm import trange

from dexterity.jacta_planner.dynamics.locomotion_plant import LocomotionPlant
from dexterity.jacta_planner.planner.action_processor import SpotWholebodyActionProcessor
from dexterity.jacta_planner.planner.parameter_container import ParameterContainer
from dexterity.jacta_planner.planner.types import ActionMode
from dexterity.jacta_planner.scenes import scene_registry
from dexterity.jacta_planner.visuals.mujoco_visualizer import MujocoRenderer

task_name = "spot_tire"

params = ParameterContainer(f"dexterity/examples/jacta_planner/config/{task_name}.yml")
params.action_time_step = 1.0 / 50.0  # polycy is 50 Hz
params.action_start_mode = ActionMode.RELATIVE_TO_CURRENT_STATE
params.model_filename = scene_registry[task_name]

plant = LocomotionPlant(params=params)
action_processor = SpotWholebodyActionProcessor(params, plant.actuated_pos)

states = params.start_state.unsqueeze(0)

renderer = MujocoRenderer(plant=plant)

render_state = states.cpu().numpy()[:, : plant.model.nq]
renderer.render(render_state)

info = {}
info["sensor"] = plant.get_sensor(states)
for _ in trange(500):
    delta_base = [1.0, 0.0, -1.0]
    delta_arm = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    relative_actions = torch.tensor([delta_base + delta_arm])

    _, _, action_trajectories = action_processor(
        relative_actions=relative_actions,
        current_states=states,
    )

    states, sensor, states_trajectory = plant.dynamics(states, action_trajectories, info)
    info["sensor"] = sensor
    for states_t in states_trajectory[0]:
        render_state = states_t.cpu().numpy()[: plant.model.nq]
        renderer.render(render_state)

renderer.play(wait_for_input=True)
