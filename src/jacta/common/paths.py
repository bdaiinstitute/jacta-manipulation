# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from pathlib import Path

from pydrake.multibody.parsing import Parser


def add_package_paths(parser: Parser) -> None:
    """Add models and data paths to the parser."""
    base_path = Path(__file__).resolve().parent.parents[1]
    parser.package_map().Add("jacta_models", str(Path(base_path, "../models")))
    parser.package_map().Add("spot_data", str(Path(base_path, "../models/meshes/spot")))
