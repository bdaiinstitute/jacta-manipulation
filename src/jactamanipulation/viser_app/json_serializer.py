# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import json
from pathlib import Path
from typing import no_type_check

import numpy as np
import torch


class ConfigEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy arrays and Path objects."""

    @no_type_check
    def default(self, obj):
        """Overrides encoding for numpy arrays and Path objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy array to list
        if isinstance(obj, Path):
            return str(obj)  # Convert Path to string
        if isinstance(obj, torch.device):
            return str(obj)
        return super().default(obj)


@no_type_check
def decode_config(obj):
    """Custom JSON decoder to handle NumPy arrays and Path strings."""
    for key, value in obj.items():
        # Convert lists of numbers back to NumPy arrays
        if isinstance(value, list) and all(isinstance(i, (int, float)) for i in value):
            obj[key] = np.array(value)
        # Convert strings that look like paths back to Path objects
        if isinstance(value, str) and ("/" in value or "\\" in value):
            obj[key] = Path(value)
    return obj
