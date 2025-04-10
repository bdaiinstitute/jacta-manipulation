import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import viser

from jacta.planner.core.parameter_container import ParameterContainer
from jacta.visualizers.mujoco.model import ViserMjModel


class TrajectoryVisualizer:
    def __init__(self, params: ParameterContainer, show_goal: bool = True):
        self.params = params
        self.server = viser.ViserServer()
        self.show_goal = show_goal

        # Set up mujoco model.
        base_path = Path(__file__).resolve().parents[4]
        self.model_path = (
            base_path / self.params.xml_folder / self.params.model_filename
        )
        assert self.model_path.exists(), f"Model file {self.model_path} not found!"
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.viser_model = ViserMjModel(self.server, self.model)
        if self.show_goal:
            self.viser_goal_model = ViserMjModel(
                self.server,
                self.model,
                show_ground_plane=False,
                alpha_scale=0.5,
                namespace="goal",
            )
            self.goal_data = mujoco.MjData(self.model)

        # Create GUI elements for controlling visualizer.
        self.running = False
        self.pause_button = self.server.gui.add_button("Start playback")
        self.timestep_slider = self.server.gui.add_slider(
            "Trajectory timestep", min=0, max=1, step=1, initial_value=0
        )
        self.playback_speed = self.server.gui.add_slider(
            "Playback speed (x realtime)", min=0.1, max=10, step=0.01, initial_value=1
        )

        # Placeholder for trajectory frames, must be set by set_trajectory.
        self.qpos = None

        @(self.pause_button).on_click
        def pb_callback(_: viser.GuiEvent) -> None:
            self.cycle_pause_button()

        @(self.timestep_slider).on_update
        def update_timestep(_: viser.GuiEvent) -> None:
            """More info about GUI callbacks in viser: https://viser.studio/versions/0.2.7/examples/03_gui_callbacks"""
            assert self.qpos is not None
            self.data.qpos[:] = self.qpos[int(self.timestep_slider.value)]
            mujoco.mj_forward(self.model, self.data)

            self.viser_model.set_data(self.data)

    def cycle_pause_button(self) -> None:
        if self.running:
            self.running = False
            self.pause_button.label = "Start playback"
        else:
            if self.timestep_slider.value == self.timestep_slider.max:
                self.timestep_slider.value = 0
            self.running = True
            self.pause_button.label = "Pause playback"

    def spin(self) -> None:
        print("Press Ctrl+C to terminate the visualizer")
        try:
            while True:
                if self.running:
                    self.timestep_slider.value = min(
                        len(self.qpos) - 1, self.timestep_slider.value + 1
                    )
                    if self.timestep_slider.value == len(self.qpos) - 1:
                        self.cycle_pause_button()

                time.sleep(self.model.opt.timestep / self.playback_speed.value)
        except KeyboardInterrupt:
            print("Closing viser...")
        finally:
            self.server.flush()
            self.server.stop()

    def set_trajectory(
        self,
        trajectory: torch.Tensor,
        goal_state: torch.Tensor | None = None,
        colors: dict[str, float] | None = None,
    ):
        trajectory = trajectory.cpu().numpy()
        qpos_inds = self.params.vis_q_indices

        # Special handling for qpos_inds edge cases.
        if type(qpos_inds) is torch.Tensor:
            qpos_inds = qpos_inds.cpu().numpy()
        elif type(qpos_inds) is None:
            qpos_inds = np.arange(trajectory.shape[1])

        self.qpos = trajectory[:, qpos_inds]
        self.timestep_slider.max = len(self.qpos) - 1
        self.timestep_slider.value = 0

        if self.show_goal and goal_state is not None:
            self.goal_data.qpos = goal_state.cpu().numpy()[: self.model.nq]
            mujoco.mj_forward(self.model, self.goal_data)
            self.viser_goal_model.set_data(self.goal_data)
