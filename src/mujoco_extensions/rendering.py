# %%
# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dexterity.jacta_planner.dynamics.mujoco_dynamics import MujocoPlant
from dexterity.jacta_planner.planner.parameter_container import ParameterContainer
from dexterity.jacta_planner.visuals.mujoco_visualizer import MujocoRenderer


# %%
def load_matrix_from_file(filename: str | Path) -> np.ndarray:
    with open(filename, "rb") as f:
        # Read dimensions
        rows = np.fromfile(f, dtype=np.int32, count=1)[0]
        cols = np.fromfile(f, dtype=np.int32, count=1)[0]
        # Read data
        data = np.fromfile(f, dtype=np.float64)
        matrix = data.reshape((rows, cols))
    return matrix


def load_matrix_from_np(filename: str) -> np.ndarray:
    return np.load(filename)


# %%
EXAMPLE_NAME = "spot_door"
EXAMPLE_NAME = "spot_ramp"
EXAMPLE_NAME = "spot_tire_rim"
# EXAMPLE_NAME = "spot_pyramid"
EXAMPLE_NAME = "spot_box"

params = ParameterContainer(f"dexterity/config/optimizer/{EXAMPLE_NAME}.yml")

# %%
plant = MujocoPlant(params=params)
renderer = MujocoRenderer(plant=plant, collision_geometry_opacity=0.0)


# %%
visualizer = renderer.visualizer
colors = {"blue_green": [0.17, 0.67, 0.59], "blue": [0.17, 0.67, 0.79], "white": [1, 1, 1], "black": [0, 0, 0]}
visualizer["/Background"].set_property("top_color", colors["white"])
visualizer["/Background"].set_property("bottom_color", colors["white"])

# %%
states = load_matrix_from_file(Path("dexterity/models") / (EXAMPLE_NAME + "_states100.bin"))

DEFAULT_COLORS = {
    "trajectory": None,
    "goal": [0.2, 0.2, 0.6, 0.5],
}
# states = np.zeros((3000, 200))
renderer.show(states[:, 0 : plant.model.nq], goal=params.goal_state, colors=DEFAULT_COLORS)


# %%
not_visible = [
    "/Axes",
    "/meshcat/goal/body",
    "/meshcat/goal/front_left_hip",
    "/meshcat/goal/front_left_upper_leg",
    "/meshcat/goal/front_left_lower_leg",
    "/meshcat/goal/front_right_hip",
    "/meshcat/goal/front_right_upper_leg",
    "/meshcat/goal/front_right_lower_leg",
    "/meshcat/goal/rear_left_hip",
    "/meshcat/goal/rear_left_upper_leg",
    "/meshcat/goal/rear_left_lower_leg",
    "/meshcat/goal/rear_right_hip",
    "/meshcat/goal/rear_right_upper_leg",
    "/meshcat/goal/rear_right_lower_leg",
    "/meshcat/goal/arm_link_sh0",
    "/meshcat/goal/arm_link_sh1",
    "/meshcat/goal/arm_link_el0",
    "/meshcat/goal/arm_link_el1",
    "/meshcat/goal/arm_link_wr0",
    "/meshcat/goal/arm_link_wr1",
    "/meshcat/goal/arm_link_fngr",
    "/meshcat/goal/door",
    "/meshcat/goal/object/rim_side",
    "/meshcat/goal/object/tire_slice_0",
    "/meshcat/goal/object/tire_slice_1",
    "/meshcat/goal/object/tire_slice_2",
    "/meshcat/goal/object/tire_slice_3",
    "/meshcat/goal/object/tire_slice_4",
    "/meshcat/goal/object/tire_slice_5",
    "/meshcat/goal/object/tire_slice_6",
    "/meshcat/goal/object/tire_slice_7",
    "/meshcat/goal/object/tire_slice_8",
    "/meshcat/goal/object/tire_slice_9",
    "/meshcat/goal/object/tire_slice_10",
    "/meshcat/goal/object/tire_slice_11",
    "/meshcat/goal/object/tire_slice_12",
    "/meshcat/goal/object/tire_slice_13",
    "/meshcat/goal/object/tire_slice_14",
    "/meshcat/goal/object/tire_slice_15",
    "/meshcat/goal/box0",
    "/meshcat/goal/box1",
    "/meshcat/goal/box2",
    "/meshcat/goal/box3",
    "/meshcat/goal/box4",
    "/meshcat/goal/box5",
    "/meshcat/trajectory/world/wall0",
    "/meshcat/trajectory/world/wall1",
    "/meshcat/trajectory/world/visual_wall0",
    "/meshcat/trajectory/world/visual_wall1",
    # "/meshcat/trajectory/world/ramp0",
    # "/meshcat/trajectory/world/ramp1",
    # # "/meshcat/trajectory/world/visual_ramp0",
    # # "/meshcat/trajectory/world/visual_ramp1",
]
for element in not_visible:
    renderer.visualizer[element].set_property("visible", False)

# %%
command = load_matrix_from_file(Path("dexterity/models") / (EXAMPLE_NAME + "_command100.bin"))
plt.plot(command[:, 0:3])
plt.show()
plt.plot(command[:, 3:10])
plt.show()
plt.plot(command[:, 10:22])
plt.show()
plt.plot(command[:, 22:25])
plt.show()

# %%
plt.plot(states[:, 0:7])
plt.show()
plt.plot(states[:, 7:19])
plt.show()
plt.plot(states[:, 19:26])
plt.show()


# %%
plt.title("base vx, vy, vz")
plt.plot((states[1:, 0:3] - states[:-1, 0:3]) / 0.01)
plt.show()
plt.title("arm velocities")
plt.plot((states[1:, 19:26] - states[:-1, 19:26]) / 0.01)
plt.show()

# %%
while True:
    time.sleep(1.0)
