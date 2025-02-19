# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import warnings
from dataclasses import Field, fields, is_dataclass
from multiprocessing.managers import DictProxy
from multiprocessing.synchronize import Event, Lock
from typing import Any, Callable, List, Literal, get_args, get_origin

import numpy as np
from viser import GuiEvent, GuiInputHandle, GuiSliderHandle, MeshHandle, ViserServer

DEFAULT_SLIDER_STEP_FLOAT = 0.01
DEFAULT_SLIDER_STEP_INT = 1
GOAL_RADIUS = 0.05


def slider(
    parameter_name: str,
    min: int | float,
    max: int | float,
    step: int | float | None = None,
) -> Callable:
    """Decorator that adds slider metadata to desired dataclass fields.

    Args:
        parameter_name: target parameter to annotate.
        min: minimum value for slider (int or float, must match dtypes of other variables).
        max: maximum value for slider (int or float).
        step: (optional) step for slider handle. If not set, defaults to constant 0.01 for floats and 1 for ints.
    """
    # Handle case where step is not specified.
    if step is None:
        if isinstance(min, int):
            step = DEFAULT_SLIDER_STEP_INT
        else:
            step = DEFAULT_SLIDER_STEP_FLOAT

    def wrapper(cls: Any) -> Any:
        """Convenience wrapper to annotate dataclass fields with slider values. Applied only to dataclasses."""
        assert is_dataclass(
            cls
        ), "@slider decorator can only be applied to dataclasses!"

        # Store original __post_init__ (to enable stacking @slider decorators , other __post_init__ behavior).
        original_post_init = getattr(cls, "__post_init__", None)

        def update_data(self: Any) -> None:
            """Callback to update field metadata after dataclass construction."""
            # Take original post_init actions, if any.
            if original_post_init:
                original_post_init(self)

            # Get the dataclass field
            class_field = cls.__dataclass_fields__[parameter_name]

            # Update the field's metadata with the provided metadata
            class_field.metadata = {"ui_config": (min, max, step)}

        # Set dataclass __post_init__ to be original + annotation step.
        cls.__post_init__ = update_data

        return cls

    return wrapper


def _get_gui_element(
    server: ViserServer, config: Any, field: Field
) -> GuiInputHandle | MeshHandle | None:
    """Helper function that creates a GUI element for a particular dataclass field.

    Args:
        server: ViserServer to which GUI elements will be added.
        config: Dataclass containing parameters for which GUI elements should be created.
        field: Current field that a GUI element will be added.
    """
    assert is_dataclass(config), "GUI elements can only be added for dataclasses!"
    init_value = getattr(config, field.name)

    # Create a slider for an integer-valued param.
    if field.type is int:
        if "ui_config" not in field.metadata:
            min_int, max_int, step_int = 1, 2 * init_value, DEFAULT_SLIDER_STEP_INT
        else:
            min_int, max_int, step_int = field.metadata["ui_config"]
        return server.gui.add_slider(
            field.name,
            min=min_int,
            max=max_int,
            step=step_int,
            initial_value=init_value,
        )

    # Create a slider for a float-valued param.
    elif field.type is float:
        if "ui_config" not in field.metadata:
            min_float, max_float, step_float = (
                0.0,
                2 * init_value,
                DEFAULT_SLIDER_STEP_FLOAT,
            )
        else:
            min_float, max_float, step_float = field.metadata["ui_config"]
        return server.gui.add_slider(
            field.name,
            min=min_float,
            max=max_float,
            step=step_float,
            initial_value=init_value,
        )

    # Create a checkbox for a boolean param.
    elif field.type is bool:
        return server.gui.add_checkbox(field.name, initial_value=init_value)

    elif field.type is np.ndarray and len(init_value) == 3:
        return server.scene.add_icosphere(
            "goal position",
            radius=GOAL_RADIUS,
            color=(0.0, 0.0, 1.0),
            position=init_value,
        )

    # Create a dropdown for a Literal-typed param.
    elif get_origin(field.type) == Literal:
        return server.gui.add_dropdown(
            field.name, get_args(field.type), initial_value=init_value
        )

    else:
        warnings.warn(
            f"Warning: Field {field.name} with type {field.type} is not a supported type for GUI creation.",
            stacklevel=1,
        )
        return None


def _get_callback(
    element: GuiInputHandle,
    element_name: str,
    config_dict: DictProxy,
    config_event: Event,
    config_lock: Lock,
) -> Callable[[GuiEvent], None]:
    """Factory method for creating a GUI callback for a particular GUI element.

    Args:
        element: GuiInputHandle that callback will be created for.
        element_name: Config dict key corresponding to GUI element.
        config_dict: multiprocessing DictProxy where config values will be written to / read from.
        config_event: Event flagging that this config param has been updated.
        config_lock: multiprocessing Lock that ensures thread safety when reading / writing config values.
    """

    def gui_callback(_: GuiEvent) -> None:
        """Updates the config_dict with the new value received from viser."""
        # Use lock for thread safety.
        with config_lock:
            if isinstance(element, GuiSliderHandle):
                # Clip numeric values to slider bounds.
                config_dict[element_name] = max(
                    min(element.value, element.max), element.min
                )
            else:
                config_dict[element_name] = element.value
        # Set config_event so other Processes know to update this value.
        config_event.set()

    return gui_callback


def create_gui_elements(
    server: ViserServer,
    config: Any,
    config_dict: DictProxy,
    config_updated_event: Event,
    config_lock: Lock,
) -> List[GuiInputHandle]:
    """Recursively iterates through a dataclass, adding GUI elements / callbacks for its fields to a Viser server.

    Args:
        server: current `ViserServer` to which the GUI elements are added.
        config: dataclass containing params which we want to update via viser.
        config_dict: `multiprocessing.DictProxy` where param values are shared between threads.
        config_updated_event: `multiprocessing.Event` that flags when config params are updated.
        config_lock: `multiprocessing.Lock` that prevents multiple processes from accessing the same param.

    Returns a list of GuiHandles that can be used to remove all elements created here on destruction.
    """
    gui_elements = []

    # Create GUI elements for each field in the dataclass.
    for config_parameter in fields(config):
        # If the dataclass is nested, traverse it recursively.
        if is_dataclass(config_parameter.type):
            config_folder = server.gui.add_folder("config_parameter.name")
            gui_elements.append(config_folder)
            with config_folder:
                new_elements = create_gui_elements(
                    server,
                    getattr(config, config_parameter.name),
                    config_dict,
                    config_updated_event,
                    config_lock,
                )
            gui_elements.append(new_elements)

        # Otherwise add an atomic GUI element for this parameter.
        else:
            element = _get_gui_element(server, config, config_parameter)

            if element is None:
                continue
            elif isinstance(element, GuiInputHandle):
                # Attach the corresponding GUI callback to update the config parameter.
                element.on_update(
                    _get_callback(
                        element,
                        config_parameter.name,
                        config_dict,
                        config_updated_event,
                        config_lock,
                    )
                )
                gui_elements.append(element)
            elif isinstance(element, MeshHandle):
                gui_elements.append(element)
    return gui_elements
