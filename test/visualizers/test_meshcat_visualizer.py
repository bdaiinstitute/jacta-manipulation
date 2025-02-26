# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import numpy as np
from pydrake.geometry import Rgba
from pydrake.trajectories import PiecewisePolynomial

from jacta.planner.core.parameter_container import ParameterContainer
from jacta.visualizers.meshcat.visuals import TrajectoryVisualizer


def test_visualize_trajectories() -> None:
    # Test visualize_trajectories():
    # making sure that the function does not error with multiple trajectories and color types.
    # Ensuring that it breaks when the number of trajectories and colors do not match.
    # In addition you can visually inspect the results through meshcat.
    example = "planar_hand"
    params = ParameterContainer()
    params.parse_params(example, "single_goal")
    visualizer = TrajectoryVisualizer(
        params=params,
        sim_time_step=0.05,
    )

    state_dim = 7
    q0_start = 0.1 * np.ones(state_dim)
    q0_end = 0.25 * np.ones(state_dim)
    q1_start = 1 * np.ones(state_dim)
    q1_end = 1.5 * np.ones(state_dim)
    q0 = np.vstack([q0_start, q0_end])
    q1 = np.vstack([q1_start, q1_end])
    q0_traj = PiecewisePolynomial.FirstOrderHold([0, 2], q0.T)
    q1_traj = PiecewisePolynomial.FirstOrderHold([0, 1], q1.T)

    # test that we need to provide as many prefixes as trajectories
    try:
        visualizer.visualize_trajectories(
            [q0_traj], ["t1", "t2"], colors={"t1": 0.3, "t2": Rgba(1, 0, 0, 0.5)}
        )
    except Exception as e:
        assert type(e) == AssertionError
    try:
        visualizer.visualize_trajectories(
            [q0_traj, q1_traj], ["t1"], colors={"t1": 0.3}
        )
    except Exception as e:
        assert type(e) == AssertionError

    # test that we need to provide transparency values between 0 and 1
    try:
        visualizer.visualize_trajectories([q0_traj], ["t1"], colors={"t1": -0.3})
    except Exception as e:
        assert type(e) == AssertionError
    try:
        visualizer.visualize_trajectories([q0_traj], ["t1"], colors={"t1": 1.3})
    except Exception as e:
        assert type(e) == AssertionError

    # This ensures that we can run the function without throwing an error.
    visualizer.visualize_trajectories([q0_traj], ["t1"], colors=None)
    visualizer.visualize_trajectories([q0_traj, q1_traj], ["t1", "t2"], colors=None)
    visualizer.visualize_trajectories(
        [q0_traj, q1_traj], ["t1", "t2"], colors={"t1": 0.0, "t2": 0.5}
    )
    visualizer.visualize_trajectories(
        [q0_traj, q1_traj],
        ["t1", "t2"],
        colors={"t1": 0.3, "t2": Rgba(1, 0.3, 0.4, 1.0)},
    )
