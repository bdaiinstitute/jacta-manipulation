# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from pathlib import Path

from jacta.common.paths import add_package_paths
from pydrake.multibody.parsing import (
    LoadModelDirectives,
    Parser,
    ProcessModelDirectives,
)
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    MultibodyPlant,
)
from pydrake.systems.framework import DiagramBuilder


def test_add_package_paths() -> None:
    # Build diagram.
    builder = DiagramBuilder()

    # MultibodyPlant
    plant = MultibodyPlant(0.01)

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, plant=plant)
    parser = Parser(plant=plant, scene_graph=scene_graph)
    add_package_paths(parser)

    base_path = Path(__file__).resolve().parent.parent
    model_directives = LoadModelDirectives(str(Path(base_path, "models/directives/planar_hand.yml")))
    ProcessModelDirectives(model_directives, plant, parser)  # type: ignore

    plant.Finalize()
    # This ensures that we can get the ground model out of the plant.
    # Otherwise this returns a RuntimeError stating that there is no
    # model instance named 'ground'.
    plant.GetModelInstanceByName("planar_hand")
