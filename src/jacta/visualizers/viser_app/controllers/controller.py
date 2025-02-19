# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


#### This base class should only contain abstract methods ###
#### Specific implementations should be made in a child-class ###
@dataclass
class ControllerConfig:
    """Base controller config with spline parameters."""


class Controller:
    """Abstract base class for all controller implementations."""

    @abstractmethod
    def make_models(self) -> None:
        """Helper to re-size the models vector to config.num_rollouts."""

    @property
    @abstractmethod
    def num_timesteps(self) -> int:
        """Helper function to recalculate the number of timesteps for simulation"""

    @property
    @abstractmethod
    def rollout_times(self) -> np.ndarray:
        """Helper function to calculate the rollout times based on the horizon length"""

    @abstractmethod
    def update_action(
        self, curr_state: np.ndarray, curr_time: float, additional_info: dict[str, Any]
    ) -> None:
        """Abstract method for updating controller actions from current state/time."""

    @abstractmethod
    def action(self, time: float) -> np.ndarray:
        """Abstract method for querying current action from controller."""

    @property
    @abstractmethod
    def controls(self) -> np.ndarray:
        """Contains the control signals applied in the current rollout."""

    @controls.setter
    @abstractmethod
    def controls(self, value: np.ndarray) -> None:
        pass

    @abstractmethod
    def set_default_controls(self) -> None:
        """Set default value for the Controller.controls. if there is no default value set to zero."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the controls and the spline to their default values."""
