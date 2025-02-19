# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import warnings
from functools import cached_property
from pathlib import Path
from typing import Any, Optional

import numpy as np  # noqa: F401
import torch
import yaml
from benedict import benedict
from torch import Tensor

from jacta.planner.core.types import (  # noqa: F401
    ActionMode,
    ClippingType,
    ControlType,
    convert_dtype,
    set_default_device_and_dtype,
)
from jacta.planner.core.types import ActionType as AT  # noqa: F401

set_default_device_and_dtype()


class ParameterContainer:
    def __init__(self) -> None:
        self._config = benedict()
        self._config.keypath_separator = "_"
        self._autofill_rules = benedict()
        self.base_path = Path(__file__).resolve().parent.parents[3]

    def __str__(self) -> str:
        return self._config.dump()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):  # handle private attributes
            return self.__dict__[name]
        return self._config[name]

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name.startswith("_"):
            super().__setattr__(__name, __value)
        else:
            self._config[__name] = __value
            if __name in self._autofill_rules:
                warnings.warn(
                    f"Parameter {__name} is being set after autofill rules have been applied."
                    "This may result in unexpected behavior."
                    f"Autofill rules are: {self._autofill_rules[__name]}",
                    stacklevel=2,
                )

    def __delattr__(self, __name: str) -> None:
        if __name.startswith("_"):
            super().__delattr__(__name)
        else:
            del self._config[__name]

    def __contains__(self, key: str) -> bool:
        return key in self._config

    def update(self, values: dict) -> None:
        if "_base_config" not in self.__dict__.keys():
            base_yml_path = Path(self.base_path, "examples/planner/config/base.yml")
            self._base_config = self.load_yaml(base_yml_path)
        vals = benedict(values)
        vals = vals.unflatten(separator="_")
        vals.keypath_separator = "_"
        self._config.merge(vals, overwrite=True)
        self.typeify()
        self.autofill()

    def parse_params(self, task: str, planner_example: str) -> None:
        self.load_base()
        self.load_task(task, planner_example)
        self.autofill()
        self.cleanup()

    def load_yaml(self, yaml_path: str) -> dict:
        temp_dict = benedict.from_yaml(yaml_path)
        temp_dict = temp_dict.unflatten(separator="_")
        temp_dict.keypath_separator = "_"
        return temp_dict

    def load_base(self) -> None:
        base_yml_path = Path(self.base_path, "examples/planner/config/base.yml")
        self._base_config = self.load_yaml(base_yml_path)
        self._config.merge(self._base_config["defaults"], overwrite=True)
        self.typeify()

    def load_task(self, task: str, planner_example: str) -> None:
        # Grab task specific config path
        if planner_example == "learning":
            task_yml_path = Path(
                self.base_path, "examples/learning/config/" + task + ".yml"
            )
        elif planner_example == "test":
            task_yml_path = Path(self.base_path, "test/config/" + task + ".yml")
        else:
            task_yml_path = Path(
                self.base_path, "examples/planner/config/task/" + task + ".yml"
            )

        self._task_config = self.load_yaml(task_yml_path)
        planner_config = self._task_config.pop("planner", {})
        planner_config.keypath_separator = "_"
        planner_config = planner_config.get(planner_example, {})

        # Overwrite base parameters with lowest level of specificity config
        self._config.merge(self._task_config, overwrite=True)
        self.typeify()
        self._config.merge(planner_config, overwrite=True)
        self.typeify()

    def autofill(self) -> None:
        # Dependencies are loaded onto class __dict__
        self._autofill_rules = self._base_config.get(
            "autofill_rules", benedict()
        ).flatten("_")

        # If top level key set, fill in dependent keys with default values
        for key in self._autofill_rules.keys():
            if key in self._config:
                self.run_autofill_rule(key)

    def run_autofill_rule(self, rule_key: str) -> None:
        for rule in self._autofill_rules[rule_key]:
            parameter_key, parameter_value = eval(rule)
            if parameter_key not in self._config or parameter_key == rule_key:
                self._config[parameter_key] = eval(parameter_value)

    def typeify(self) -> None:
        # Assign types before loading into class dict
        for keypath in self._config.keypaths():
            if isinstance(self._config[keypath], str) and "file" not in keypath:
                self._config[keypath] = convert_dtype(eval(self._config[keypath]))
            if "distribution" in keypath:
                self._config[keypath] /= torch.sum(self._config[keypath])
            elif "indices" in keypath:
                self._config[keypath] = convert_dtype(
                    self._config[keypath], dtype=torch.int64
                )
            elif "proximity_scaling" in keypath:
                value = self._config[keypath]
                self._config[keypath] = (
                    torch.tensor([value]) if isinstance(value, float) else value
                )

    def reset_seed(self) -> None:
        self.set_seed(self.seed)

    def set_seed(self, value: Optional[int]) -> None:
        self.seed = value
        if value is not None:
            torch.manual_seed(value)

    @cached_property
    def reward_distance_scaling_sqrt(self) -> Tensor:
        return torch.sqrt(self.reward_distance_scaling)

    @property
    def xml_folder(self) -> Path:
        return Path(self.base_path, "models/xml")

    def cleanup(self) -> None:
        del self._base_config
        del self._task_config

        # Set final parameters
        self.reset_seed()


def parse_hardware_parameters(file_path: str) -> dict:
    with open(file_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)
            return dict()
