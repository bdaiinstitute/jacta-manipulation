# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.


import mujoco
import numpy as np
from viser import ViserServer

from jacta.visualizers.mujoco.model import ViserMjModel

viser_server = ViserServer()

model_path = "models/xml/box_push.xml"
model = mujoco.MjModel.from_xml_path(model_path)


def test_model_loading() -> None:
    viser_model = ViserMjModel(viser_server, model)
    assert viser_model is not None, "Failed to create ViserMjModel"
    assert len(viser_model._bodies) == model.nbody, "Incorrect number of bodies"
    assert len(viser_model._geoms) > 0, "No geometries created"


def test_set_data() -> None:
    viser_model = ViserMjModel(viser_server, model)
    data = mujoco.MjData(model)
    data.qpos = np.random.randn(model.nq)
    data.qvel = np.random.randn(model.nv)
    mujoco.mj_forward(model, data)
    viser_model.set_data(data)
    for i in range(1, len(viser_model._bodies)):
        assert np.allclose(viser_model._bodies[i].position, data.xpos[i]), f"Position mismatch for body {i}"
        assert np.allclose(viser_model._bodies[i].wxyz, data.xquat[i]), f"Orientation mismatch for body {i}"


def test_ground_plane() -> None:
    viser_model_with_plane = ViserMjModel(viser_server, model, show_ground_plane=True)
    assert any("ground_plane" in geom.name for geom in viser_model_with_plane._geoms), "Ground plane not found"

    viser_model_no_plane = ViserMjModel(viser_server, model, show_ground_plane=False)
    assert all("ground_plane" not in geom.name for geom in viser_model_no_plane._geoms), "Unexpected ground plane"


# TODO(pculbert): Add more unit tests for loading each element type from XML.

if __name__ == "__main__":
    test_model_loading()
    test_set_data()
    test_ground_plane()