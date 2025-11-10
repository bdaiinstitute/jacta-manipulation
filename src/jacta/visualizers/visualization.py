# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
from abc import abstractmethod
from typing import Protocol


class Visualization(Protocol):
    """Container for updating the viser model with current state info."""

    @abstractmethod
    def _update_stats(self) -> None:
        """Adds text with the statistics of performance in the GUI."""

    @abstractmethod
    def update_visualization(self) -> None:
        """Update model state and traces."""

    @abstractmethod
    def remove(self) -> None:
        """Removes all model geometries from the GUI."""
