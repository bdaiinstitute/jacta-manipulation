# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import mujoco


class MujocoPlant:
    """Simple convenience object containing the model and data."""

    def __init__(self, xml_model_path: str) -> None:
        """Constructs a MujocoPlant object."""
        self.model = mujoco.MjModel.from_xml_path(xml_model_path)
        self.data = mujoco.MjData(self.model)
