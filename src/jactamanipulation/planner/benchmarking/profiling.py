# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
"""Allows the user to profile planner instances."""

import importlib.util
import sys
from pathlib import Path


def run_example(example_type: str, example_name: str, test_mode: bool = True) -> None:
    """Profiles any of the planners' performance on a given task.

    Args:
        example_type: can be "planner_single_goal", "planner_multi_goal", "planner_exploration".
        example_name: can be "floating_hand", "proximity_optimization", "allegro_hand".
        test_mode: Running in test mode this is much faster but won't fully complete the task.
    """
    example = [example_type, example_name]
    print("testing example:", *example)
    spec = None
    if example[0] in ["planner_single_goal", "planner_multi_goal", "planner_exploration"]:
        spec = importlib.util.spec_from_file_location(
            "examples." + example[1],
            Path("dexterity/examples") / example[0] / "example.py",
        )
    else:
        spec = importlib.util.spec_from_file_location(
            "examples." + example[1],
            Path("dexterity/examples") / example[0] / (example[1] + ".py"),
        )
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules["examples." + example[1]] = module
        setattr(module, "test", test_mode)  # noqa: B010
        setattr(module, "example", example[1])  # noqa: B010
        spec.loader.exec_module(module)
    print("finished example:", *example)
