# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import logging
from dataclasses import dataclass, field
from typing import Any

import mink
import mujoco
import mujoco.viewer
import numpy as np


@dataclass
class IkSolver:
    """Simplistic inverse kinematics solver"""

    model: mujoco.MjModel
    frame_names: list[str] = field(
        default_factory=lambda: [
            "site_arm_link_fngr",
            "site_front_left",
            "site_front_right",
            "site_rear_left",
            "site_rear_right",
        ]
    )
    frame_type: str = "site"
    solver: str = "quadprog"
    position_cost: np.ndarray = field(default_factory=lambda: np.array([1, 1, 1, 1, 1]))
    orientation_cost: np.ndarray = field(
        default_factory=lambda: 1e-1 * np.array([1, 0, 0, 0, 0])
    )
    posture_cost: np.ndarray = field(
        default_factory=lambda: 1e-2 * np.array([1, 1, 20, 10, 10, 1] + [1] * 19)
    )
    pos_threshold: float = 1e-2
    ori_threshold: float = 1.0
    max_iters: int = 20
    dt: float = 0.002
    damping: float = 1e-3
    frame_tasks_damping: np.ndarray = field(
        default_factory=lambda: np.array([1, 1, 1, 1, 1])
    )
    # Private fields for lazy initialization
    _frame_tasks: dict[str, Any] = field(default_factory=dict, init=False)
    _posture_task: Any = field(init=False)
    _configuration: Any = field(init=False)
    _configuration_limit: Any = field(init=False)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    def __post_init__(self) -> None:
        """Initialize tasks and configuration after the dataclass is initialized."""
        assert len(self.position_cost) == len(
            self.frame_names
        ), "position_cost must have the same length as frame_names"
        assert len(self.orientation_cost) == len(
            self.frame_names
        ), "orientation_cost must have the same length as frame_names"
        assert len(self.frame_tasks_damping) == len(
            self.frame_names
        ), "frame_tasks_damping must have the same length as frame_names"

        self._initialize_tasks()
        self._initialize_configuration()

    def _initialize_tasks(self) -> None:
        """Initialize the end effector and posture tasks."""
        self.frame_tasks = []
        for frame_name, position_cost, orientation_cost, damping in zip(
            self.frame_names,
            self.position_cost,
            self.orientation_cost,
            self.frame_tasks_damping,
            strict=False,
        ):
            setattr(
                self,
                f"{frame_name}_task",
                mink.FrameTask(
                    frame_name=frame_name,
                    frame_type=self.frame_type,
                    position_cost=position_cost,
                    orientation_cost=orientation_cost,
                    lm_damping=damping,
                ),
            )
            self.frame_tasks.append(getattr(self, f"{frame_name}_task"))

        self._posture_task = mink.PostureTask(
            model=self.model, cost=self.posture_cost[: self.model.nv]
        )

    def _initialize_configuration(self) -> None:
        """Initialize the robot's configuration model."""
        self._configuration = mink.Configuration(self.model)
        self._configuration_limit = mink.ConfigurationLimit(self.model)

    @property
    def posture_task(self) -> Any:
        """Posture task"""
        return self._posture_task

    @property
    def configuration(self) -> Any:
        """Configuration"""
        return self._configuration

    @property
    def configuration_limit(self) -> Any:
        """Configuration limits"""
        return self._configuration_limit

    def set_target_pose(self, target_poses: list[np.ndarray | None]) -> None:
        """Set the target pose for the frame tasks."""
        for frame_task, pose in zip(self.frame_tasks, target_poses, strict=False):
            if pose is not None:
                rotation = mink.SO3(pose[3:7])
                translation = pose[0:3]
                target_transform = mink.SE3.from_rotation_and_translation(
                    rotation, translation
                )
                frame_task.set_target(target_transform)

    def solve(
        self,
        q_ref: np.ndarray,
        target_poses: list[np.ndarray | None],
        use_configuration_limit: bool = True,
        logging_level: int = logging.INFO,
    ) -> np.ndarray:
        """Solve inverse kinematics to reach the target pose."""
        assert not all(
            pose is None for pose in target_poses
        ), "At least need one target pose."

        if len(target_poses) != len(self.frame_tasks):
            logging.warning(
                f"Number of target poses {len(target_poses)} does not match "
                f"the number of frame tasks {len(self.frame_tasks)}. Adding None to the end of target_poses."
            )
            target_poses = [target_poses[0]] + [None] * (
                len(self.frame_tasks) - len(target_poses)
            )

        if self.model.nv == 13:  # No leg model from spot_mink.xml
            assert target_poses[0] is not None, (
                "You are using the spot model without legs, you need to specify the first target_pose for "
                "the end effector."
            )
            target_poses = [target_poses[0]] + [None] * (len(self.frame_tasks) - 1)

        # Set the initial configuration
        self.configuration.update(q_ref)
        self.posture_task.set_target_from_configuration(self.configuration)

        # Set target pose for frame tasks
        self.set_target_pose(target_poses)

        # Adjust logging level
        logging.getLogger().setLevel(logging_level)

        # Create a list of tasks based on whether the target pose is provided
        tasks = [self.posture_task]
        for frame_task, pose in zip(self.frame_tasks, target_poses, strict=False):
            if pose is not None:
                tasks.append(frame_task)

        limits = [self.configuration_limit] if use_configuration_limit else []

        # Solve IK iteratively
        for i in range(self.max_iters):
            velocity = mink.solve_ik(
                self.configuration,
                tasks,
                self.dt,
                self.solver,
                limits=limits,
                damping=self.damping,
            )
            self.configuration.integrate_inplace(velocity, self.dt)

            # Compute error for each frame task
            pos_errs, ori_errs = [], []
            for frame_task, pose in zip(self.frame_tasks, target_poses, strict=False):
                if pose is not None:
                    error = frame_task.compute_error(self.configuration)
                    pos_err = np.linalg.norm(error[:3]) * np.sign(
                        frame_task.position_cost
                    )
                    ori_err = np.linalg.norm(error[3:]) * np.sign(
                        frame_task.orientation_cost
                    )
                    pos_errs.append(pos_err)
                    ori_errs.append(ori_err)

                    # Print iteration information
                    logging.debug(
                        f"Iteration {i} Frame {frame_task.frame_name}: "
                        f"Position Error = {pos_err:.2f}, Orientation Error = {ori_err:.2f}"
                    )

            # Check if errors are within the acceptable thresholds
            if all([pos_err <= self.pos_threshold for pos_err in pos_errs]) and all(
                [ori_err <= self.ori_threshold for ori_err in ori_errs]
            ):
                logging.debug(f"Converged after {i} iterations.")
                break
        else:
            logging.warning(
                f"Reached maximum iterations ({self.max_iters}) without convergence."
            )

        return self.configuration.q
