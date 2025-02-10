# %%
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import numpy as np

from dexterity.jacta_planner.dynamics.mujoco_dynamics import MujocoPlant
from dexterity.jacta_planner.planner.parameter_container import ParameterContainer
from dexterity.jacta_planner.visuals.mujoco_visualizer import MujocoRenderer

# %%
num_waypoints = 5
log_time_step = 0.05

# %%
params = ParameterContainer("dexterity/examples/jacta_planner/config/spot_floating_box.yml")
plant = MujocoPlant(params=params)

renderer = MujocoRenderer(plant=plant, time_step=log_time_step)

# trajectory
times = np.linspace(0, (num_waypoints - 1) * 0.4, num_waypoints)
q0 = 0.0 * np.array(
    [
        1.0,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
)

state_size = params.start_state.shape[0]
q_ref = np.array([0.2 * (i + 1) * np.ones(state_size) for i in range(num_waypoints)])
# %%
renderer.show(q_ref)

input("Press Enter to continue...")
