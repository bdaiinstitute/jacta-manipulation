# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.path_resolvers import get_starfish_path
from jactamanipulation.planner.scenes.scene_composer import Scene, make_default_header_includes

ASSET_DIR = Path("dexterity/models/meshes/")
XML_DIR = Path("dexterity/models/xml/")
COMPONENTS_DIR = XML_DIR / "components/"
OUTPUT_DIR = XML_DIR / "scenes/generated"


@dataclass
class SceneRegistry:
    """Registry for MuJoCo scenes."""

    def __post_init__(self) -> None:
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._scenes: dict[str, dict[str, Any]] = {}

    def add(self, name: str, **scene_kwargs: Any) -> None:
        """Register a new scene."""
        self._scenes[name] = scene_kwargs

    def generate(self, name: str) -> None:
        """Generate XML for a specific scene."""
        if name not in self._scenes:
            raise KeyError(f"Scene '{name}' is not registered")

        scene_kwargs = self._scenes[name]
        header_includes = make_default_header_includes(COMPONENTS_DIR, ASSET_DIR)
        scene = Scene(**scene_kwargs, header_includes=header_includes)

        # Generate XML file - outside the source tree
        scene.to_xml_file(get_starfish_path() / self.output_dir / f"{name}.xml")

    def generate_all(self) -> None:
        """Generate XML for all registered scenes."""
        for name in self._scenes:
            self.generate(name)

    def __getitem__(self, name: str) -> Path:
        """Get the path to a scene's XML file."""
        if name not in self._scenes:
            raise KeyError(f"Scene '{name}' is not registered")
        return OUTPUT_DIR / f"{name}.xml"


# Global registry instance
scene_registry = SceneRegistry()
