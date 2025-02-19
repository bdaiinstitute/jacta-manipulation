# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

# Code modified from https://github.com/amacati/dextgen, MIT license
"""Normalizer module.

Enables distributed normalizers to keep preprocessing consistent across all nodes. The normalizer is
based on the implementation in https://github.com/openai/baselines.
"""

from io import BufferedWriter
from typing import Any

import torch
from torch import FloatTensor


class Normalizer:
    """Normalizer to maintain an estimate on the current data mean and variance.

    Used to normalize input data to zero mean and unit variance.
    """

    def __init__(self, size: int, eps: float = 1e-2) -> None:
        """Initialize local and global buffer arrays for distributed mode.

        Args:
            size: Data dimension. Each dimensions mean and variance is tracked individually.
            eps: Minimum variance value to ensure numeric stability. Has to be larger than 0.
        """
        self.eps2 = torch.ones(size, dtype=torch.float32) * eps**2
        self.sum = torch.zeros(size, dtype=torch.float32)
        self.sum_sq = torch.zeros(size, dtype=torch.float32)
        self.count = 0
        self.mean = torch.zeros(size, dtype=torch.float32)
        self.std = torch.ones(size, dtype=torch.float32)

    def __call__(self, x: FloatTensor) -> FloatTensor:
        """Alias for `self.normalize`."""
        return self.normalize(x)

    def normalize(self, x: FloatTensor) -> FloatTensor:
        """Normalize the input data with the current mean and variance estimate.

        Args:
            x: Input data array.

        Returns:
            The normalized data.
        """
        norm = (x - self.mean) / self.std
        return norm

    def update(self, x: FloatTensor) -> None:
        """Update the mean and variance estimate with new data.

        Args:
            x: New input data. Expects a 3D array of shape (episodes, timestep, data dimension).

        Raises:
            AssertionError: Shape check failed.
        """
        assert x.ndim != 3, "Expecting 3D arrays of shape (episodes, timestep, data dimension)!"

        self.sum += torch.sum(x, axis=0, dtype=torch.float32)
        self.sum_sq += torch.sum(x**2, axis=0, dtype=torch.float32)
        self.count += x.shape[0]

        self.mean = self.sum / self.count
        self.std = self.sum_sq / self.count - (self.sum / self.count) ** 2
        torch.maximum(self.eps2, self.std, out=self.std)  # Numeric stability
        torch.sqrt(self.std, out=self.std)

    def wrap_obs(self, states: FloatTensor, goals: FloatTensor) -> FloatTensor:
        """Wrap states and goals into a contingent input tensor.

        Args:
            states: Input states array.
            goals: Input goals array.

        Returns:
            A fused state goal tensor.
        """
        states, goals = self.normalize(states), self.normalize(goals)
        return torch.concatenate((states, goals), axis=states.ndim - 1)

    def load(self, checkpoint: Any) -> None:
        """Load data for the state_norm.

        Args:
            checkpoint: dict containing loaded data.
        """
        self.eps2 = checkpoint["eps2"]
        self.sum = checkpoint["sum"]
        self.sum_sq = checkpoint["sum_sq"]
        self.count = checkpoint["count"]
        self.mean = checkpoint["mean"]
        self.std = checkpoint["std"]

    def save(self, f: BufferedWriter) -> None:
        """Save data for the state_norm."""
        torch.save(
            {
                "eps2": self.eps2,
                "sum": self.sum,
                "sum_sq": self.sum_sq,
                "count": self.count,
                "mean": self.mean,
                "std": self.std,
            },
            f,
        )
