# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

"""Class for rendering Mujoco trajectories in meshcat"""

from typing import List

import numpy as np
from mujoco import MjModel


def get_body_name(model: MjModel, bodyid: int) -> str:
    """Return name of the body with given ID from MjModel."""
    index = model.name_bodyadr[bodyid]
    end = model.names.find(b"\x00", index)
    name = model.names[index:end].decode("utf-8")
    if len(name) == 0:
        name = f"body{bodyid}"
    return name


def get_geom_name(model: MjModel, geomid: int) -> str:
    """Return name of the geom with given ID from MjModel."""
    index = model.name_geomadr[geomid]
    end = model.names.find(b"\x00", index)
    name = model.names[index:end].decode("utf-8")
    if len(name) == 0:
        name = f"geom{geomid}"
    return name


def get_geometry_rgba(model: MjModel, geomid: int) -> List[float]:
    """Return RGBA color for geom with given ID from MjModel."""
    geom_matid = model.geom_matid[geomid]
    geom_rgba = np.array(model.geom_rgba[geomid])
    mat_rgba = np.array(model.mat_rgba[geom_matid])
    default_mat_rgba = np.array([0.1, 0.1, 0.1, 1.0])
    default_geom_rgba = np.array([0.5, 0.5, 0.5, 1.0])
    if np.allclose(mat_rgba, default_mat_rgba) and not np.allclose(geom_rgba, default_geom_rgba):
        return geom_rgba.tolist()
    else:
        return mat_rgba.tolist()


def get_mesh_data(model: MjModel, meshid: int) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve the vertices and faces of a specified mesh from a MuJoCo model.

    Args:
        model : MjModel The MuJoCo model containing the mesh data.
        meshid : int The index of the mesh to retrieve.

    Result:
        tuple[np.ndarray, np.ndarray]
        Vertices (N, 3) and faces (M, 3) of the mesh.
    """
    vertadr = model.mesh_vertadr[meshid]
    vertnum = model.mesh_vertnum[meshid]
    vertices = model.mesh_vert[vertadr : vertadr + vertnum, :]

    faceadr = model.mesh_faceadr[meshid]
    facenum = model.mesh_facenum[meshid]
    faces = model.mesh_face[faceadr : faceadr + facenum]
    return vertices, faces
