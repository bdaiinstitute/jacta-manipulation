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

from jactamanipulation.planner.planner.action_processor import (  # noqa: F401
    ActionProcessor,
    SpotFloatingActionProcessor,
    SpotWholebodyActionProcessor,
)
from jactamanipulation.planner.planner.linear_algebra import gram_schmidt
from jactamanipulation.planner.planner.planner_helpers import (  # noqa: F401
    is_object_out_of_reach,
    is_object_tilted,
    torso_too_close_to_object,
)
from jactamanipulation.planner.planner.types import (  # noqa: F401
    ActionMode,
    ClippingType,
    ControlType,
    convert_dtype,
    set_default_device_and_dtype,
)
from jactamanipulation.planner.planner.types import ActionType as AT  # noqa: F401

set_default_device_and_dtype()


class ParameterContainer:
    """Parameter container class"""

    def __init__(
        self,
        yaml_path: Path | str | None = None,
        base_yaml_path: Path = Path("base.yml"),
    ) -> None:
        self._config = benedict()
        self._config.keypath_separator = "_"
        self._autofill_rules = benedict()
        if isinstance(yaml_path, str):
            yaml_path = Path(yaml_path)
        if yaml_path is not None:
            self._parse_params(yaml_path, base_yaml_path)

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
        """Update

        Args:
            values (dict): Values to update with
        """
        vals = benedict(values)
        vals = vals.unflatten(separator="_")
        vals.keypath_separator = "_"
        self._config.merge(vals, overwrite=True)
        self._typeify()
        self.autofill()

    def _parse_params(self, yaml_path: Path, base_yaml_path: Path) -> None:
        """Parse parameters

        Args:
            yaml_path (Path): Path to yaml file
            base_yaml_path (Path): Path to base yaml file
        """
        self.load_base(base_yaml_path)
        self._load_task(yaml_path)
        self.autofill()
        self._cleanup()

    def _load_yaml(self, yaml_path: Path) -> dict:
        """Load from yaml

        Args:
            yaml_path (Path): Path of yaml file to load from

        Returns:
            dict: Values in yaml
        """
        temp_dict = benedict.from_yaml(yaml_path)
        temp_dict = temp_dict.unflatten(separator="_")
        temp_dict.keypath_separator = "_"
        return temp_dict

    def load_base(self, base_yaml_path: Path = Path("base.yml")) -> None:
        """Load base

        Args:
            base_yaml_path (Path): Path to base yaml file
        """
        self.base_yaml_path = base_yaml_path
        base_yml_path = self.examples_directory / "planner/config" / self.base_yaml_path
        self._base_config = self._load_yaml(base_yml_path)
        self._config.merge(self._base_config["defaults"], overwrite=True)
        self._typeify()

    def _load_task(self, yaml_path: Path) -> None:
        """Load task

        Args:
            yaml_path (Path): Path to yaml file
        """
        self._task_config = self._load_yaml(yaml_path)
        planner_config = self._task_config.pop("planner", {})
        planner_config.keypath_separator = "_"

        # Overwrite base parameters with lowest level of specificity config
        self._config.merge(self._task_config, overwrite=True)
        self._typeify()
        self._config.merge(planner_config, overwrite=True)
        self._typeify()

    def autofill(self) -> None:
        """Autofill"""
        # Dependencies are loaded onto class __dict__
        self._autofill_rules = self._base_config.get("autofill_rules", benedict()).flatten("_")

        # If top level key set, fill in dependent keys with default values
        for key in self._autofill_rules.keys():
            if key in self._config:
                self.run_autofill_rule(key)

    def run_autofill_rule(self, rule_key: str) -> None:
        """Run autofill rule

        Args:
            rule_key (str): Rule key
        """
        for rule in self._autofill_rules[rule_key]:
            parameter_key, parameter_value = eval(rule)
            if parameter_key not in self._config or parameter_key == rule_key:
                self._config[parameter_key] = eval(parameter_value)

    def list_of_yaml_arrays_to_tensor(self, lists: list) -> Tensor:
        """Convert list of lists to tensor"""
        vec = []
        for item in lists:
            if isinstance(item, float) or isinstance(item, int):
                vec.append(item)
            elif isinstance(item, list):
                vec.extend(item)
            elif isinstance(item, str):
                vec.extend(eval(item))
            else:
                raise ValueError("Invalid type in list of arrays")

        return convert_dtype(vec)

    def concat_list_of_arrays(self, keypath: str) -> Tensor:
        """Concat list of arrays

        Args:
            keypath (str): Keypath

        Returns:
            Tensor: Arrays
        """
        arrays = self._config[keypath]
        if isinstance(arrays, float):
            return torch.tensor([arrays])
        if not isinstance(arrays, Tensor):
            arrays = self.list_of_yaml_arrays_to_tensor(arrays)
        return arrays

    def process_scaling(self, keypath: str) -> Tensor:
        """Process scaling

        Args:
            keypath (str): keypath

        Returns:
            Tensor: Scaling
        """
        scaling = self.concat_list_of_arrays(keypath)
        if "distance" in keypath and scaling.ndim == 1:
            scaling = torch.diag(scaling)
        return scaling

    def _typeify(self) -> None:
        """Typeify"""
        no_eval_keys = ["file", "name", "skill", "yaml"]
        # Assign types before loading into class dict
        for keypath in self._config.keypaths():
            if isinstance(self._config[keypath], str) and all(key not in keypath for key in no_eval_keys):
                self._config[keypath] = convert_dtype(eval(self._config[keypath]))
            if "distribution" in keypath:
                self._config[keypath] /= torch.sum(self._config[keypath])
            elif "indices" in keypath:
                self._config[keypath] = convert_dtype(self._config[keypath], dtype=torch.int64)
            elif keypath.endswith(("_scaling", "_thresholds")):
                self._config[keypath] = self.process_scaling(keypath)
            elif keypath.startswith(("start_", "goal_")) and keypath.endswith(("state", "_lower", "_upper")):
                self._config[keypath] = self.concat_list_of_arrays(keypath)
            elif keypath.endswith("_fns"):
                self._config[keypath] = [eval(fn) if isinstance(fn, str) else fn for fn in self._config[keypath]]
            elif isinstance(self._config[keypath], list):
                self._config[keypath] = convert_dtype(self._config[keypath])

    def load_eigenspaces(self) -> None:
        """Load eigenspaces"""
        if self.using_eigenspaces and "eigenspaces_file" in self._config:
            eigenspace_file = self.model_folder / "eigenspaces" / self.eigenspaces_file
            with open(eigenspace_file, "r") as eigen_file:
                parsed_yaml = yaml.safe_load(eigen_file)
                self.basis_vectors = parsed_yaml["eigenspaces"]
            self.orthonormal_basis = gram_schmidt(torch.tensor(self.basis_vectors))

    def reset_seed(self) -> None:
        """Reset seed"""
        self.set_seed(self.seed)

    def set_seed(self, value: Optional[int]) -> None:
        """Set seed

        Args:
            value (Optional[int]): Seed value
        """
        self.seed = value
        if value is not None and value >= 0:
            torch.manual_seed(value)

    @cached_property
    def reward_distance_scaling_sqrt(self) -> Tensor:
        """Reward distance scaling sqrt

        Returns:
            Tensor: Scaling sqrt
        """
        return torch.sqrt(self.reward_distance_scaling)

    @property
    def model_folder(self) -> Path:
        """Model folder

        Returns:
            Path: Path to models folder
        """
        return Path("dexterity/models")

    @property
    def examples_directory(self) -> Path:
        """Examples directory

        Returns:
            Path: Path to examples directory
        """
        return Path("dexterity/examples")

    @property
    def xml_folder(self) -> Path:
        """XML folder path

        Returns:
            Path: Path to xml folder
        """
        return self.model_folder / "xml"

    @property
    def data_folder(self) -> Path:
        """Data folder

        Returns:
            Path: Data folder
        """
        return Path("dexterity/data")

    @property
    def cache_folder(self) -> Path:
        """Cache folder

        Returns:
            Path: Path to cache folder
        """
        return self.data_folder / "cache"

    @property
    def state_cache_folder(self) -> Path:
        """State cache folder

        Returns:
            Path: Path to state cache folder
        """
        return self.cache_folder / "states"

    @property
    def graph_cache_folder(self) -> Path:
        """Graph cache folder

        Returns:
            Path: Path to graph cache folder
        """
        return self.cache_folder / "graphs"

    @property
    def trajectory_cache_folder(self) -> Path:
        """Trajectory cache folder

        Returns:
            Path: Path to trajectory cache folder
        """
        return self.cache_folder / "trajectories"

    @property
    def policy_filepath(self) -> str:
        """Policy filepath

        Returns:
            str: Path to policy file
        """
        return str(self.data_folder / "policies" / self.policy_filename)

    @property
    def device(self) -> torch.device:
        """Device

        Returns:
            torch.device: Tensor device
        """
        dummy_tensor = torch.tensor([0])
        return dummy_tensor.device

    def _cleanup(self) -> None:
        """Cleanup"""
        # Set final parameters
        self.reset_seed()
        self.load_eigenspaces()


def parse_hardware_parameters(file_path: str) -> dict:
    """Parse hardware parameters from file

    Args:
        file_path (str): Filepath

    Returns:
        dict: Parameters from file
    """
    with open(file_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)
            return dict()
