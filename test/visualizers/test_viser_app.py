import mujoco
import numpy as np
from trimesh.creation import box
from viser import ViserServer

# Import the functions and classes from the module under test.
from jacta.visualizers.mujoco.model import (
    ViserMjModel,
    add_box,
    add_capsule,
    add_cylinder,
    add_ellipsoid,
    add_mesh,
    add_plane,
    add_segments,
    add_sphere,
    add_spline,
    rgba_float_to_int,
    rgba_int_to_float,
    set_mesh_color,
    set_segment_points,
    set_spline_positions,
)

# Create a global ViserServer instance for use by the tests.
viser_server = ViserServer()

# Use an existing model file for tests that require a valid MjModel.
# (Make sure that the path "models/xml/box_push.xml" is correct in your environment.)
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
        assert np.allclose(
            viser_model._bodies[i].position, data.xpos[i]
        ), f"Position mismatch for body {i}"
        assert np.allclose(
            viser_model._bodies[i].wxyz, data.xquat[i]
        ), f"Orientation mismatch for body {i}"


def test_ground_plane() -> None:
    viser_model_with_plane = ViserMjModel(viser_server, model, show_ground_plane=True)
    assert any(
        "ground_plane" in geom.name for geom in viser_model_with_plane._geoms
    ), "Ground plane not found"

    viser_model_no_plane = ViserMjModel(viser_server, model, show_ground_plane=False)
    assert all(
        "ground_plane" not in geom.name for geom in viser_model_no_plane._geoms
    ), "Unexpected ground plane"


def test_rgba_conversion() -> None:
    rgba_float = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    rgba_int = rgba_float_to_int(rgba_float)
    expected_int = (255 * rgba_float).astype(int)
    assert np.array_equal(
        rgba_int, expected_int
    ), "rgba_float_to_int did not scale correctly"
    rgba_float_converted = rgba_int_to_float(rgba_int)
    # Allow slight differences due to rounding.
    assert np.allclose(
        rgba_float, rgba_float_converted, atol=1 / 255.0
    ), "rgba_int_to_float did not convert correctly"


def test_set_mesh_color() -> None:
    mesh = box([1, 1, 1])
    rgba = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    set_mesh_color(mesh, rgba)
    expected_int = rgba_float_to_int(rgba)
    # Check a few face colors (all should be set to the same integer color)
    assert np.array_equal(
        mesh.visual.material.main_color, expected_int
    ), "set_mesh_color did not set the color correctly"


def test_add_plane() -> None:
    plane = add_plane(
        viser_server,
        "test_plane",
        pos=np.array([1, 2, 3]),
        quat=np.array([1, 0, 0, 0]),
    )
    assert "test_plane" in plane.name, "add_plane did not set correct name"


def test_add_sphere() -> None:
    sphere = add_sphere(
        viser_server,
        "test_sphere",
        radius=1.0,
        pos=np.zeros(3),
        quat=np.array([1, 0, 0, 0]),
        rgba=np.array([0.5, 0.5, 0.5, 1.0]),
    )
    assert "test_sphere" in sphere.name, "add_sphere did not set correct name"


def test_add_cylinder() -> None:
    cylinder_geom = add_cylinder(
        viser_server,
        "test_cylinder",
        radius=1.0,
        height=2.0,
        pos=np.ones(3),
        quat=np.array([1, 0, 0, 0]),
        rgba=np.array([0.5, 0.5, 0.5, 1.0]),
    )
    assert (
        "test_cylinder" in cylinder_geom.name
    ), "add_cylinder did not set correct name"


def test_add_box() -> None:
    box_geom = add_box(
        viser_server,
        "test_box",
        size=np.array([1.0, 1.0, 1.0]),
        pos=np.array([0, 0, 0]),
        quat=np.array([1, 0, 0, 0]),
        rgba=np.array([0.2, 0.3, 0.4, 1.0]),
    )
    assert "test_box" in box_geom.name, "add_box did not set correct name"


def test_add_capsule() -> None:
    capsule_geom = add_capsule(
        viser_server,
        "test_capsule",
        radius=0.5,
        length=2.0,
        pos=np.array([0, 0, 0]),
        quat=np.array([1, 0, 0, 0]),
        rgba=np.array([0.3, 0.4, 0.5, 1.0]),
    )
    assert "test_capsule" in capsule_geom.name, "add_capsule did not set correct name"


def test_add_ellipsoid() -> None:
    scaling = np.array([1.0, 2.0, 3.0])
    ellipsoid_geom = add_ellipsoid(
        viser_server,
        "test_ellipsoid",
        scaling=scaling,
        pos=np.array([0, 0, 0]),
        quat=np.array([1, 0, 0, 0]),
        rgba=np.array([0.6, 0.7, 0.8, 1.0]),
    )
    assert (
        "test_ellipsoid" in ellipsoid_geom.name
    ), "add_ellipsoid did not set correct name"


def test_add_mesh() -> None:
    mesh = box([1, 1, 1])
    vertices = mesh.vertices
    faces = mesh.faces
    mesh_geom = add_mesh(
        viser_server,
        "test_mesh",
        vertices=vertices,
        faces=faces,
        pos=np.array([0, 0, 0]),
        quat=np.array([1, 0, 0, 0]),
        rgba=np.array([0.4, 0.5, 0.6, 1.0]),
    )
    assert "test_mesh" in mesh_geom.name, "add_mesh did not set correct name"


def test_add_spline() -> None:
    positions = ((0, 0, 0), (1, 1, 1), (2, 2, 2))
    spline_handle = add_spline(
        viser_server,
        "test_spline",
        positions=positions,
        pos=np.array([0, 0, 0]),
        quat=np.array([1, 0, 0, 0]),
        rgb=(0.1, 0.2, 0.3),
    )
    assert "test_spline" in spline_handle.name, "add_spline did not set correct name"


def test_add_segments() -> None:
    points = np.random.rand(4, 2, 3)
    segments_handle = add_segments(
        viser_server,
        "test_segments",
        points=points,
        pos=np.zeros(3),
        quat=np.array([1, 0, 0, 0]),
        rgb=(0.2, 0.3, 0.4),
    )
    assert (
        "test_segments" in segments_handle.name
    ), "add_segments did not set correct name"


# Dummy classes to test setter helpers that do not require a full ViserServer instance.


class DummySpline:
    def __init__(self):
        self.positions = None


def test_set_spline_positions() -> None:
    dummy = DummySpline()
    positions_tuple = ((0, 0, 0), (1, 1, 1), (2, 2, 2))
    set_spline_positions(dummy, positions_tuple)
    assert (
        dummy.positions == positions_tuple
    ), "set_spline_positions did not set positions correctly"
    # Also test with numpy array input.
    positions_np = np.array(positions_tuple)
    set_spline_positions(dummy, positions_np)
    expected = tuple(map(tuple, positions_np))
    assert (
        dummy.positions == expected
    ), "set_spline_positions did not convert numpy array correctly"


class DummySegment:
    def __init__(self):
        self.points = None


def test_set_segment_points() -> None:
    dummy = DummySegment()
    points = np.random.rand(5, 2, 3)
    set_segment_points(dummy, points)
    np.testing.assert_allclose(
        dummy.points, points, err_msg="set_segment_points did not set points correctly"
    )


def test_remove_traces() -> None:
    viser_model = ViserMjModel(viser_server, model)
    # Ensure _traces is defined.
    assert hasattr(viser_model, "_traces"), "Model does not have _traces attribute"
    # Remove the traces.
    viser_model.remove_traces()
    assert viser_model._traces == [], "remove_traces did not clear the traces list"


def test_set_traces() -> None:
    viser_model = ViserMjModel(viser_server, model)
    # Create a dummy trace array with shape (3, 2, 3)
    traces = np.random.rand(3, 2, 3)
    all_traces_rollout_size = 2
    viser_model.set_traces(traces, all_traces_rollout_size=all_traces_rollout_size)
    # Check that two trace handles have been created.
    assert (
        len(viser_model._traces) == 2
    ), "set_traces did not create 2 trace handles when needed"
    # Check that the points in the 'best' trace handle equal traces[:2]
    np.testing.assert_allclose(
        viser_model._traces[0].points,
        traces[:all_traces_rollout_size],
        err_msg="Best trace points not set correctly",
    )
    # Check that the points in the 'other' trace handle equal traces[2:]
    np.testing.assert_allclose(
        viser_model._traces[1].points,
        traces[all_traces_rollout_size:],
        err_msg="Other trace points not set correctly",
    )
    # Now test with None input to remove traces.
    viser_model.set_traces(None, all_traces_rollout_size=all_traces_rollout_size)
    assert viser_model._traces == [], "set_traces with None did not remove traces"


def test_remove() -> None:
    viser_model = ViserMjModel(viser_server, model)
    viser_model.remove()
    # Verify that traces are cleared.
    assert viser_model._traces == [], "remove did not clear _traces"
    # Note: geometries are removed via their remove() method; further checking
    # would require inspection of ViserServer's scene state.


if __name__ == "__main__":
    # Run all test functions.
    test_model_loading()
    test_set_data()
    test_ground_plane()
    test_rgba_conversion()
    test_set_mesh_color()
    test_add_plane()
    test_add_sphere()
    test_add_cylinder()
    test_add_box()
    test_add_capsule()
    test_add_ellipsoid()
    test_add_mesh()
    test_add_spline()
    test_add_segments()
    test_set_spline_positions()
    test_set_segment_points()
    test_remove_traces()
    test_set_traces()
    test_remove()
    print("All tests passed.")
