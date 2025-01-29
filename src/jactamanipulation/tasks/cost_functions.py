# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import numpy as np


def smooth_l1_norm(input: np.ndarray, p: float) -> np.ndarray:
    return np.sqrt(np.power(input, 2) + p**2) - p


def quadratic_norm(input: np.ndarray) -> np.ndarray:
    return 0.5 * np.square(input).sum(-1)
