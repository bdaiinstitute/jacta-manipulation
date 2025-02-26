# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
from pydrake.geometry import (
    Meshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Rgba,
    Role,
    RoleAssign,
    SceneGraph,
    StartMeshcat,
)
from pydrake.multibody.meshcat import JointSliders
from pydrake.multibody.parsing import (
    LoadModelDirectives,
    ModelDirectives,
    Parser,
    ProcessModelDirectives,
)
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import ModelInstanceIndex
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import TrajectorySource
from pydrake.systems.rendering import MultibodyPositionToGeometryPose
from pydrake.trajectories import PiecewisePolynomial

from jacta.common.paths import add_package_paths
from jacta.planner.core.parameter_container import ParameterContainer

DEFAULT_PREFIXES = ["actual", "desired"]
DEFAULT_COLORS = {
    "actual": 1.0,
    "desired": Rgba(0.8, 0.2, 0.2, 0.2),
}


def visualize_and_control_model(
    meshcat: Meshcat,
    directives_filename: str,
    initial_joint_values: Optional[npt.ArrayLike] = None,
    lower_joint_limits: Optional[npt.ArrayLike] = None,
    upper_joint_limits: Optional[npt.ArrayLike] = None,
) -> None:
    """Creates a visualization of a meshcat object with sliders attached to manipulate the joints."""
    builder = DiagramBuilder()
    scene_graph = builder.AddSystem(SceneGraph())

    # Parse the plant
    plant = MultibodyPlant(time_step=0.0)
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    parser = Parser(plant, scene_graph)
    add_package_paths(parser)

    # Load the models with the given plant and parameters
    base_path = Path(__file__).resolve().parent.parents[3]
    directives_path = Path(base_path, "models/directives", directives_filename)
    model_directives = LoadModelDirectives(directives_path)
    _ = ProcessModelDirectives(
        model_directives,
        plant,
        parser,
    )

    # Finish parsing the plant
    plant.Finalize()
    nq = plant.num_positions()
    plant.SetDefaultPositions(np.zeros(nq))

    # Connect the sliders to the input of the pose builder
    to_pose = builder.AddSystem(MultibodyPositionToGeometryPose(plant))
    builder.Connect(
        to_pose.get_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id()),
    )
    sliders = builder.AddSystem(
        JointSliders(
            meshcat,
            plant,
            initial_value=initial_joint_values,
            lower_limit=lower_joint_limits,
            upper_limit=upper_joint_limits,
        )
    )
    builder.Connect(sliders.get_output_port(), to_pose.get_input_port(0))

    # Initialize the meshcat visualization parameters
    meshcat_params = MeshcatVisualizerParams()
    meshcat_params.delete_on_initialization_event = False
    meshcat_params.role = Role.kIllustration
    _ = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, meshcat_params)

    diagram = builder.Build()

    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    print("Use the slider in the MeshCat controls to change the joint positions.")
    print("Press 'Stop Simulation' in MeshCat to continue.")
    meshcat.AddButton("Stop Simulation")
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 1.0)
    meshcat.DeleteAddedControls()


class TrajectoryVisualizer:
    def __init__(
        self,
        params: ParameterContainer,
        sim_time_step: float,
    ) -> None:
        self.traj_systems: List[TrajectorySource] = []
        self._meshcat = StartMeshcat()
        self._params = params
        self.sim_time_step = sim_time_step
        base_path = Path(__file__).resolve().parent.parents[3]
        self._directives_path = str(
            Path(base_path, "models/directives", self._params.vis_filename)
        )

    @property
    def meshcat(self) -> Meshcat:
        return self._meshcat

    def set_directives_prefix(
        self,
        model_directives: ModelDirectives,
        prefix: str = "directive::",
    ) -> None:
        """Add a prefix to rename the model directives.
        This function supports the "add_model" and "add_weld" directives.

        Args:
            model_directives: model directives to be renamed.
            prefix: the string that will be added in front of the names of all directives.

        """
        for model_directive in model_directives.directives:
            model = model_directive.add_model
            weld = model_directive.add_weld
            if model is not None:
                model.name = prefix + model.name
            elif weld is not None:
                weld.child = prefix + weld.child
                if weld.parent != "world":
                    weld.parent = prefix + weld.parent
            else:
                print(
                    "Unsupported directives: add_model_instance, add_frame, add_collision_filter_group, add_directives"
                )

    def set_color_of_models(
        self,
        plant: MultibodyPlant,
        model_instances: List[ModelInstanceIndex],
        scene_graph: SceneGraph,
        color: Optional[Union[Rgba, float]] = None,
    ) -> None:
        """Set the color and transparency of the given model instances.

        Args:
            plant: the multibody plant that contains the model instances.
            model_instances: the model instance that will be recolored.
            scene_graph: the scene graph to be modified.
            color: the new color and transparency as Rgba or just the transparency as a float (between 0 and 1),
            if we want to keep the original colors (rgb) of the bodies.
        """

        inspector = scene_graph.model_inspector()
        for model in model_instances:
            for body_id in plant.GetBodyIndices(model):
                frame_id = plant.GetBodyFrameIdOrThrow(body_id)
                for geometry_id in inspector.GetGeometries(
                    frame_id, Role.kIllustration
                ):
                    properties = inspector.GetIllustrationProperties(geometry_id)
                    phong = properties.GetPropertyOrDefault(
                        "phong", "diffuse", Rgba(0, 0, 0, 0.0)
                    )
                    if color is None:
                        phong.set(phong.r(), phong.g(), phong.b(), phong.a())
                    elif isinstance(color, Rgba):
                        phong.set(color.r(), color.g(), color.b(), color.a())
                    else:
                        assert 0.0 <= color <= 1.0
                        phong.set(phong.r(), phong.g(), phong.b(), color)
                    properties.UpdateProperty("phong", "diffuse", phong)
                    scene_graph.AssignRole(
                        plant.get_source_id(),
                        geometry_id,
                        properties,
                        RoleAssign.kReplace,
                    )

    def _initialize_visualization_environment(
        self,
        prefixes: List[str],
        colors: Optional[Dict[str, Union[float, Rgba]]],
    ) -> None:
        self._scene_graph = SceneGraph()
        self._scene_graph.set_name("scene_graph")

        self.plants = []
        for prefix in prefixes:
            plant = MultibodyPlant(time_step=0.0)
            plant.RegisterAsSourceForSceneGraph(self._scene_graph)
            parser = Parser(plant, self._scene_graph)
            add_package_paths(parser)

            model_directives = LoadModelDirectives(self._directives_path)
            self.set_directives_prefix(model_directives, prefix=prefix + "::")
            model_instances = ProcessModelDirectives(
                model_directives,
                plant,
                parser,
            )

            plant.Finalize()
            nq = plant.num_positions()
            plant.SetDefaultPositions(np.zeros(nq))

            color = None if colors is None else colors[prefix]
            self.set_color_of_models(
                plant,
                [m.model_instance for m in model_instances],
                self._scene_graph,
                color=color,
            )

            self.plants.append(plant)

    def setup_and_connect(
        self, trajectories: List[PiecewisePolynomial.FirstOrderHold]
    ) -> None:
        self._builder = DiagramBuilder()
        self._builder.AddSystem(self._scene_graph)

        for i, traj in enumerate(trajectories):
            to_pose = self._builder.AddSystem(
                MultibodyPositionToGeometryPose(self.plants[i])
            )
            self._builder.Connect(
                to_pose.get_output_port(),
                self._scene_graph.get_source_pose_port(self.plants[i].get_source_id()),
            )

            traj_system = self._builder.AddSystem(TrajectorySource(traj))
            self._builder.Connect(
                traj_system.get_output_port(), to_pose.get_input_port(0)
            )
            self.traj_systems.append(traj_system)

        meshcat_params = MeshcatVisualizerParams()
        meshcat_params.delete_on_initialization_event = False
        meshcat_params.role = Role.kIllustration
        self.visualizer = MeshcatVisualizer.AddToBuilder(
            self._builder, self._scene_graph, self._meshcat, meshcat_params
        )
        self._meshcat.Delete()
        self.diagram = self._builder.Build()

    def update_trajectories(
        self, trajectories: List[PiecewisePolynomial.FirstOrderHold]
    ) -> None:
        # Update the trajectories
        for i, traj in enumerate(trajectories):
            traj_system = self.traj_systems[i]
            traj_system.UpdateTrajectory(traj)

    def visualize_trajectories(
        self,
        trajectories: List[PiecewisePolynomial.FirstOrderHold],
        prefixes: List[str],
        colors: Optional[Dict[str, Union[float, Rgba]]] = None,
    ) -> None:
        assert len(prefixes) == len(trajectories)
        t_final = np.max(np.array([traj.end_time() for traj in trajectories]))

        if len(self.traj_systems) != len(trajectories):
            self._initialize_visualization_environment(prefixes, colors)
            self.setup_and_connect(trajectories)
        else:
            self.update_trajectories(trajectories)

        self._simulator = Simulator(self.diagram)
        self.visualizer.StartRecording()
        self._simulator.AdvanceTo(t_final)
        self.visualizer.PublishRecording()

    def show(
        self,
        trajectory: torch.FloatTensor,
        goal_state: Optional[torch.FloatTensor] = None,
        colors: Optional[Dict[str, Union[float, Rgba]]] = DEFAULT_COLORS,
    ) -> None:
        trajectory = trajectory.cpu().numpy()
        if trajectory.shape[0] == 1:
            trajectory = np.stack((trajectory[0], trajectory[0]))

        num_sim_steps = trajectory.shape[0] - 1
        times = np.linspace(0, num_sim_steps * self.sim_time_step, num_sim_steps + 1)

        vis_q_indices = self._params.vis_q_indices
        if type(vis_q_indices) is torch.Tensor:
            vis_q_indices = vis_q_indices.cpu().numpy()
        if vis_q_indices is None:
            vis_q_indices = np.arange(trajectory.shape[1])
        q_trajectory = trajectory[:, vis_q_indices].T

        trajectory = PiecewisePolynomial.FirstOrderHold(times, q_trajectory)

        trajectories = [trajectory]
        prefixes = ["actual"]

        if goal_state is not None:
            goal_state = goal_state.cpu().numpy()
            # reference trajectory
            times = [0, self.sim_time_step]
            q_ref = np.tile(goal_state[vis_q_indices], (2, 1)).T
            ref_trajectory = PiecewisePolynomial.FirstOrderHold(times, q_ref)
            trajectories.append(ref_trajectory)
            prefixes.append("desired")

        # visualization
        self.visualize_trajectories(trajectories, prefixes, colors)
