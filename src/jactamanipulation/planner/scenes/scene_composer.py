# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import mujoco


# TODO: remove when python is bumped to 3.12 - https://docs.python.org/3/library/typing.html#typing.override
def override(f: Callable) -> Callable:
    return f


@dataclass
class mjBodyWrapper:
    """Wrapper for MuJoCo body objects that provides simplified child body attachment and attribute access."""

    body: mujoco.MjsBody

    def attach(self, child_body: mujoco.MjsBody, name: str | None = None) -> None:
        """Attach a child body to this body using a frame."""
        frame = self.body.add_frame()

        # add prefix - important for copying the <defaults> (prevents repeated entries)
        prefix = f"{name}/" if name else f"{child_body.name}/"
        frame.attach_body(child_body, prefix, "")

        # simplify the name of the attached body (removes the prefix)
        new_name = f"{name}" if name else f"{child_body.name}"
        self.body.find_child(f"{prefix}{child_body.name}").name = new_name

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped MjsBody instance."""
        return getattr(self.body, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Delegate attribute setting to the wrapped MjsBody instance, except for the 'body' attribute."""
        if name == "body":
            # Special case for setting the 'body' attribute during initialization
            super().__setattr__(name, value)
        else:
            setattr(self.body, name, value)


@dataclass
class ModelElement:
    """Base class for MuJoCo model elements that handles XML loading, includes management, and body access."""

    name: str
    xml_content: str = "<mujoco><worldbody></worldbody></mujoco>"
    header_includes: dict = field(default_factory=dict)

    @classmethod
    def from_file(cls, name: str, xml_path: Path, header_includes: dict | None = None) -> "ModelElement":
        """Load an XML file and create a ModelElement."""
        xml_content = xml_path.read_text()
        return cls(name, xml_content, header_includes or {})

    @classmethod
    def from_string(cls, name: str, xml_content: str, header_includes: dict | None = None) -> "ModelElement":
        """Create a ModelElement from an XML string."""
        return cls(name, xml_content, header_includes or {})

    @staticmethod
    def autoload(f: Callable) -> Callable:
        """Decorator to ensure the spec is loaded before executing the decorated function."""

        def wrapper(self: "ModelElement", *args: Any, **kwargs: Any) -> Any:
            if not hasattr(self, "_spec"):
                # warnings.warn(f"Model {self.name!r} was not loaded. Auto-loading now.", stacklevel=3)
                self.load()
            return f(self, *args, **kwargs)

        return wrapper

    def load(self, header_includes: dict | None = None) -> None:
        """Load/initialize the model element with parameters.

        Args:
            name: Name for the model element
            header_includes: Dictionary of include files to use
        """
        if header_includes:
            self.header_includes.update(header_includes)

        # Initialize the model
        self._spec = self._load_spec()

    @autoload
    def __getitem__(self, key: str) -> mjBodyWrapper:
        """Get the body of this model element based on the body name."""
        body = self._spec.find_body(key)
        if body is None:
            raise ValueError(f"Body {key} not found in model {self.name}")
        return mjBodyWrapper(body)

    @property
    def body(self) -> mjBodyWrapper:
        """Get the main body of this model element based on the model name."""
        return self[self.name]

    @autoload
    def to_xml_string(self) -> str:
        """Generate XML string from MjSpec."""
        self._spec.compile()
        xml = self._spec.to_xml()

        # Edit the XML string to update mesh and texture directories
        xml = xml.replace('meshdir="', 'meshdir="../../../../../')
        xml = xml.replace('texturedir="', 'texturedir="../../../../../')

        return xml

    def to_xml_file(self, path: Path) -> None:
        """Write XML string to file."""
        with open(path, "w") as f:
            f.write(self.to_xml_string())

    def _load_spec(self) -> mujoco.MjSpec:
        """Load XML with proper include handling."""
        xml_str = self._load_xml_string()
        includes = self._load_includes()
        return mujoco.MjSpec().from_string(xml_str, includes)

    def _load_xml_string(self) -> str:
        """Load XML string from file or string. Ensure worldbody exists."""
        xml = self.xml_content

        # Ensure worldbody exists
        if "<worldbody>" not in xml:
            start, end = xml.find("<body"), xml.rfind("</body>") + 7
            if -1 < start < end:
                prefix, bodies, suffix = xml[:start], xml[start:end], xml[end:]
                xml = f"{prefix}<worldbody>{bodies}</worldbody>{suffix}"

        # Generate includes header from header_includes
        if self.header_includes:
            includes_header = "\n".join(f'  <include file="{filename}" />' for filename in self.header_includes.keys())
            # Add includes header before worldbody
            xml = re.sub(r"(<worldbody[^>]*>)", rf"{includes_header}\g<1>", xml, flags=re.DOTALL)

        return xml.rstrip()

    def _load_includes(self) -> dict[str, bytes]:
        """Load and encode include files."""
        includes = {}
        for name, content in self.header_includes.items():
            # Read content if it's a Path
            content = content.read_text() if isinstance(content, Path) else content
            # Convert to UTF-8 bytes
            includes[name] = content.encode("utf-8")
        return includes


@dataclass
class SpotRobot(ModelElement):
    """Model element representing a Spot robot with base, legs, and arm components."""

    name: str = "spot"
    xml_dir: Path = field(default_factory=lambda: Path("dexterity/models/xml/components/"))

    @override
    def load(self, header_includes: dict | None = None) -> None:
        """Load the Spot robot model element."""
        super().load(header_includes)
        ############### Add base body ###################
        base = ModelElement.from_file(
            name="base",
            xml_path=self.xml_dir / "spot" / "base_body.xml",
            header_includes={
                **self.header_includes,
                "spot/base_defs.xml": self.xml_dir / "spot" / "base_defs.xml",
            },
        )
        # attach base to the world
        self["world"].attach(base.body)

        ############### Add legs ########################
        legs = ModelElement.from_file(
            name="legs",
            xml_path=self.xml_dir / "spot" / "legs.xml",
            header_includes={
                **self.header_includes,
                "spot/_legs_defs.xml": self.xml_dir / "spot" / "legs_defs.xml",
            },
        )
        # attach legs to the base
        self["base"].attach(legs.body)

        ############### Add arm #########################
        arm = ModelElement.from_file(
            name="arm",
            xml_path=self.xml_dir / "spot" / "arm.xml",
            header_includes={
                **self.header_includes,
                "spot/_arm_defs.xml": self.xml_dir / "spot" / "arm_defs.xml",
            },
        )
        # attach arm to the base
        self["base"].attach(arm.body)

    @property
    @override
    def body(self) -> mjBodyWrapper:
        """Get the main body of this model element based on the model name."""
        return self["base"]


@dataclass
class Scene(ModelElement):
    """Model element representing a complete MuJoCo scene composed of multiple model elements."""

    name: str = "scene"
    elements: dict = field(default_factory=dict)
    sensors: list = field(default_factory=list)
    contacts: list = field(default_factory=list)
    xml_dir: Path = field(default_factory=lambda: Path("dexterity/models/xml/components/"))

    def _load_element_from_path(self, path: str) -> ModelElement:
        """Load a model element from a path."""
        name = path.split("/")[-1]  # get the last element of the path (e.g. "common/ground" -> "ground")

        header_includes = self.header_includes
        defs_path = self.xml_dir / f"{path}_defs.xml"
        if defs_path.exists():  # if the defs file exists, add it to the header includes
            header_includes[f"{path}_defs.xml"] = defs_path
        return ModelElement.from_file(name, self.xml_dir / f"{path}.xml", header_includes)

    def _add_sensor(self, **sensor_kwargs: str) -> None:
        """Add a sensor to the scene.

        Args:
            **sensor_kwargs: Sensor arguments (objname for sensor type, frameyaxis, objname and refname for framepos)

        Note:
            Currently supported sensor types are:
            - framepos
            - frameyaxis
        """
        sensor = self._spec.add_sensor()
        sensor.name = f"{self.name}_{sensor_kwargs['name']}"

        if sensor_kwargs["type"] == "framepos":
            sensor.type = mujoco.mjtSensor.mjSENS_FRAMEPOS
            sensor.objtype = mujoco.mjtObj.mjOBJ_SITE
            sensor.objname = sensor_kwargs["objname"]

            if "refname" in sensor_kwargs:
                refname = sensor_kwargs["refname"]
                sensor.reftype = mujoco.mjtObj.mjOBJ_SITE
                sensor.refname = refname

        elif sensor_kwargs["type"] == "frameyaxis":
            sensor.type = mujoco.mjtSensor.mjSENS_FRAMEYAXIS
            sensor.objtype = mujoco.mjtObj.mjOBJ_SITE
            sensor.objname = sensor_kwargs["objname"]

        else:
            raise ValueError(f"Invalid sensor type: {sensor_kwargs['type']}")

    def _add_contact(self, **contact_kwargs: str) -> None:
        """Add a contact exclusion between two bodies.

        Args:
            contact_kwargs: Contact arguments (body1 and body2)
        """
        exclude = self._spec.add_exclude()
        exclude.bodyname1 = contact_kwargs["body1"]
        exclude.bodyname2 = contact_kwargs["body2"]

    @override
    def load(self, header_includes: dict | None = None) -> None:
        """Load the scene model element."""
        super().load(header_includes)

        # Load scene elements
        for name, element in self.elements.items():
            if isinstance(element, str):
                element = self._load_element_from_path(element)
            element.load(self.header_includes)
            self["world"].attach(element.body, name)

        # Add sensors
        for sensor_spec in self.sensors:
            self._add_sensor(**sensor_spec)

        # Add contacts
        for contact_spec in self.contacts:
            self._add_contact(**contact_spec)


def make_default_header_includes(xml_dir: Path, asset_dir: Path) -> dict:
    header_includes = {}

    # Add compiler.xml
    compiler_content = (xml_dir / "spot" / "compiler.xml").read_text()
    compiler_content = re.sub(r'assetdir="[^"]*"', f'assetdir="{asset_dir}"', compiler_content)
    header_includes["compiler.xml"] = compiler_content

    # Add spot's defs.xml
    header_includes["spot/defs.xml"] = (xml_dir / "spot" / "defs.xml").read_text()

    return header_includes
