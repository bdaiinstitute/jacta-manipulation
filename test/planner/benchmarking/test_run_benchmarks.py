# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
import torch
from torch import tensor

from dexterity.planner.benchmarking.benchmarking import Benchmark
from dexterity.planner.dynamics.mujoco_dynamics import MujocoPlant
from dexterity.planner.planner.action_sampler import ActionSampler
from dexterity.planner.planner.graph import Graph
from dexterity.planner.planner.graph_worker import SingleGoalWorker
from dexterity.planner.planner.logger import Logger
from dexterity.planner.planner.parameter_container import ParameterContainer
from dexterity.planner.planner.planner import Planner
from dexterity.planner.planner.types import ActionType as AT


def check_progress(benchmark: Benchmark) -> None:
    if not all(benchmark.results_distance < 1.0):  # retry once
        print("First benchmark run failed. Continuing search once.")
        benchmark.run_benchmark()
        benchmark.print_results()
    assert all(benchmark.results_distance < 1.0), "Some planners have not made any progress"


def floating_hand_setup_benchmark() -> Benchmark:
    a_min = tensor([-2, -2, -torch.pi / 2, -2.4, 0.0, 0.0])
    a_max = tensor([+2, +2, 0.0, 0.0, torch.pi / 2, 2.4])
    a_range = tensor([0.25, 0.25, 2.5, 2.5, 2.5, 2.5])

    distance_scaling = torch.diag(
        torch.cat((tensor([1, 1, 1 / (2 * torch.pi)]), torch.zeros(6), torch.zeros(3), torch.zeros(6)))
    )

    start_state = tensor([0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    goal_orientations = tensor([0, 1, 2, 3, 4]) * torch.pi / 2
    planners = []

    for orientation in goal_orientations:
        goal_state = tensor([0.1, 0.4, orientation, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        params = ParameterContainer()
        params.load_base()
        params.model_filename = ("bimanual_station.xml",)
        params.vis_filename = ("bimanual_station.yml",)
        params.start_state = (start_state,)
        params.goal_state = (goal_state,)
        params.reward_distance_scaling = (distance_scaling,)
        params.action_bound_lower = (a_min,)
        params.action_bound_upper = (a_max,)
        params.action_range = (a_range,)
        params.steps_per_goal = (1000,)
        params.autofill()

        plant = MujocoPlant(params=params)
        graph = Graph(plant, params)
        logger = Logger(graph, params)
        action_sampler = ActionSampler(plant, graph, params)
        graph_worker = SingleGoalWorker(plant, graph, action_sampler, logger, params)

        planner = Planner(plant, graph, action_sampler, graph_worker, logger, params)
        planners.append(planner)

    names = ["0", "pi/2", "pi", "3pi/2", "2pi"]
    benchmark = Benchmark(planners, names=names, verbose=False)

    return benchmark


def kuka_setup_benchmark() -> Benchmark:
    a_min = -1.0 * tensor(
        [
            2.81706,
            1.9444,
            2.81706,
            1.9444,
            2.81706,
            1.9444,
            2.90433,
            2.81706,
            1.9444,
            2.81706,
            1.9444,
            2.81706,
            1.9444,
            2.90433,
        ]
    )
    a_max = -a_min
    a_range = torch.ones_like(a_min) * 2.5

    distance_scaling = torch.zeros((40, 40))

    start_box_pose = tensor([0.6, 0, 0.251, 1, 0, 0, 0])
    arm_state = tensor(
        [0, 0.8, 0, -1.6, 0, 0, 0, 0, 0.8, 0, -1.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    start_state = torch.cat((start_box_pose, arm_state))

    goal_orientations = tensor([1, 2]) * torch.pi / 2
    planners = []

    # Rotate around x-axis
    for theta in goal_orientations:
        distance_scaling = torch.diag(torch.cat((torch.ones(6), torch.zeros(14), torch.zeros(20))))
        goal_box_pose = tensor([0.6, 0, 0.251, torch.cos(theta / 2), torch.sin(theta / 2), 0, 0])
        goal_state = torch.cat((goal_box_pose, arm_state))

        params = ParameterContainer()
        params.load_base()
        params.model_filename = ("bimanual_station.xml",)
        params.vis_filename = ("bimanual_station.yml",)
        params.start_state = (start_state,)
        params.goal_state = (goal_state,)
        params.reward_distance_scaling = (distance_scaling,)
        params.action_bound_lower = (a_min,)
        params.action_bound_upper = (a_max,)
        params.action_range = (a_range,)
        params.steps_per_goal = (1000,)
        params.autofill()

        plant = MujocoPlant(params=params)
        graph = Graph(plant, params)
        logger = Logger(graph, params)
        action_sampler = ActionSampler(plant, graph, params)
        graph_worker = SingleGoalWorker(plant, graph, action_sampler, logger, params)

        planner = Planner(plant, graph, action_sampler, graph_worker, logger, params)
        planners.append(planner)

    # Rotate around y-axis
    for theta in goal_orientations:
        distance_scaling = torch.diag(torch.cat((torch.ones(6), torch.zeros(14), torch.zeros(20))))
        goal_box_pose = tensor([0.6, 0, 0.251, torch.cos(theta / 2), 0, torch.sin(theta / 2), 0])
        goal_state = torch.cat((goal_box_pose, arm_state))

        params = ParameterContainer()
        params.load_base()
        params.model_filename = ("bimanual_station.xml",)
        params.vis_filename = ("bimanual_station.yml",)
        params.start_state = (start_state,)
        params.goal_state = (goal_state,)
        params.reward_distance_scaling = (distance_scaling,)
        params.action_bound_lower = (a_min,)
        params.action_bound_upper = (a_max,)
        params.action_range = (a_range,)
        params.autofill()

        plant = MujocoPlant(params=params)
        graph = Graph(plant, params)
        logger = Logger(graph, params)
        action_sampler = ActionSampler(plant, graph, params)
        graph_worker = SingleGoalWorker(plant, graph, action_sampler, logger, params)

        planner = Planner(plant, graph, action_sampler, graph_worker, logger, params)
        planners.append(planner)

    # Rotate around z-axis
    for theta in goal_orientations:
        distance_scaling = torch.diag(torch.cat((torch.ones(6), torch.zeros(14), torch.zeros(20))))
        goal_box_pose = tensor([0.6, 0, 0.251, torch.cos(theta / 2), 0, 0, torch.sin(theta / 2)])
        goal_state = torch.cat((goal_box_pose, arm_state))

        params = ParameterContainer()
        params.load_base()
        params.model_filename = ("bimanual_station.xml",)
        params.vis_filename = ("bimanual_station.yml",)
        params.start_state = (start_state,)
        params.goal_state = (goal_state,)
        params.reward_distance_scaling = (distance_scaling,)
        params.action_bound_lower = (a_min,)
        params.action_bound_upper = (a_max,)
        params.action_range = (a_range,)
        params.autofill()

        plant = MujocoPlant(params=params)
        graph = Graph(plant, params)
        logger = Logger(graph, params)
        action_sampler = ActionSampler(plant, graph, params)
        graph_worker = SingleGoalWorker(plant, graph, action_sampler, logger, params)

        planner = Planner(plant, graph, action_sampler, graph_worker, logger, params)
        planners.append(planner)

    # push
    distance_scaling = torch.diag(torch.cat((torch.ones(6), torch.zeros(14), torch.zeros(20))))
    goal_box_pose = tensor([0.8, 0.2, 0.251, 1, 0, 0, 0])
    goal_state = torch.cat((goal_box_pose, arm_state))

    params = ParameterContainer()
    params.load_base()
    params.model_filename = ("bimanual_station.xml",)
    params.vis_filename = ("bimanual_station.yml",)
    params.start_state = (start_state,)
    params.goal_state = (goal_state,)
    params.reward_distance_scaling = (distance_scaling,)
    params.action_bound_lower = (a_min,)
    params.action_bound_upper = (a_max,)
    params.action_range = (a_range,)
    params.autofill()

    plant = MujocoPlant(params=params)
    graph = Graph(plant, params)
    logger = Logger(graph, params)
    action_sampler = ActionSampler(plant, graph, params)
    graph_worker = SingleGoalWorker(plant, graph, action_sampler, logger, params)

    planner = Planner(plant, graph, action_sampler, graph_worker, logger, params)
    planners.append(planner)

    # lift
    distance_scaling = torch.diag(torch.cat((tensor([1, 1, 5, 1, 1, 1]), torch.zeros(14), torch.zeros(20))))
    goal_box_pose = tensor([0.6, 0.0, 0.451, 1, 0, 0, 0])
    goal_state = torch.cat((goal_box_pose, arm_state))

    params = ParameterContainer()
    params.load_base()
    params.model_filename = ("bimanual_station.xml",)
    params.vis_filename = ("bimanual_station.yml",)
    params.start_state = (start_state,)
    params.goal_state = (goal_state,)
    params.reward_distance_scaling = (distance_scaling,)
    params.action_bound_lower = (a_min,)
    params.action_bound_upper = (a_max,)
    params.action_range = (a_range,)
    params.autofill()

    plant = MujocoPlant(params=params)
    graph = Graph(plant, params)
    logger = Logger(graph, params)
    action_sampler = ActionSampler(plant, graph, params)
    graph_worker = SingleGoalWorker(plant, graph, action_sampler, logger, params)

    planner = Planner(plant, graph, action_sampler, graph_worker, logger, params)
    planners.append(planner)

    names = ["x pi/2", "x pi", "y pi/2", "y pi", "z pi/2", "z pi", "push", "lift"]
    benchmark = Benchmark(planners, names=names, verbose=False)

    return benchmark


def planar_hand_setup_benchmark() -> Benchmark:
    a_min = tensor([-torch.pi / 2, -2.4, 0.0, 0.0])
    a_max = tensor([0.0, 0.0, torch.pi / 2, 2.4])
    a_range = torch.ones_like(a_min) * 2.5

    distance_scaling = torch.diag(
        torch.cat((tensor([1, 1, 1 / (2 * torch.pi)]), torch.zeros(4), torch.zeros(3), torch.zeros(4)))
    )

    start_state = tensor([0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    goal_orientations = tensor([0, 1, 2, 3, 4]) * torch.pi / 2
    planners = []

    for orientation in goal_orientations:
        goal_state = tensor([0.0, 0.3, orientation, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        params = ParameterContainer()
        params.load_base()
        params.model_filename = "dexterity/models/xml/scenes/legacy/planar_hand.xml"
        params.start_state = start_state
        params.goal_state = goal_state
        params.reward_distance_scaling = distance_scaling
        params.action_bound_lower = a_min
        params.action_bound_upper = a_max
        params.action_range = a_range
        params.steps_per_goal = 2
        params.autofill()

        plant = MujocoPlant(params=params)
        graph = Graph(plant, params)
        logger = Logger(graph, params)
        action_sampler = ActionSampler(plant, graph, params)
        graph_worker = SingleGoalWorker(plant, graph, action_sampler, logger, params)

        planner = Planner(plant, graph, action_sampler, graph_worker, logger, params)
        planners.append(planner)

    names = ["0", "pi/2", "pi", "3pi/2", "2pi"]
    benchmark = Benchmark(planners, names=names, verbose=False)

    return benchmark


def spot_setup_benchmark() -> Benchmark:
    a_min = tensor([-2, -2, -2, -2.62, -torch.pi, 0, -2.79, -1.83, -2.87, -1.57])
    a_max = tensor([+2, +2, +2, torch.pi, 0.52, torch.pi, 2.79, 1.83, 2.87, 0])
    a_range = torch.ones_like(a_min) * 2.5

    distance_scaling = torch.diag(
        torch.cat((1.0 * torch.ones(6), 0.0 * torch.ones(10), 0.0 * torch.ones(6), 0.0 * torch.ones(10)))
    )

    initial_velocity = torch.zeros(16)

    start_box_pose = tensor([0, 0, 0.275, 1, 0, 0, 0])
    goal_box_pose = tensor([1.0, 0, 0.275, 1, 0, 0, 0])
    robot_pose = tensor([-1.0, 0.0, 0.2, 0, -2, +2, 0, 0, 0, -1])

    start_state = torch.cat((start_box_pose, robot_pose, initial_velocity))
    goal_state = torch.cat((goal_box_pose, robot_pose, initial_velocity))

    params = ParameterContainer(
        model_filename="spot_floating.xml",
        vis_filename="spot_floating.yml",
        start_state=start_state,
        goal_state=goal_state,
        distance_scaling=distance_scaling,
        action_bound_lower=a_min,
        action_bound_upper=a_max,
        action_range=a_range,
        proximity_scaling=0.1 * torch.ones(24),
        action_types=[AT.RANGED, AT.PROXIMITY],
        action_distribution=[0.5, 0.5],
        log_to_file=False,
        steps_per_goal=1000,
    )

    plant = MujocoPlant(params=params)
    graph = Graph(plant, params)
    logger = Logger(graph, params)
    action_sampler = ActionSampler(plant, graph, params)
    graph_worker = SingleGoalWorker(plant, graph, action_sampler, logger, params)

    planner = Planner(plant, graph, action_sampler, graph_worker, logger, params)

    planners = [planner]
    names = ["spot"]
    benchmark = Benchmark(planners, names=names, verbose=False)

    return benchmark


# Just checks that the benchmark is running, not actually benchmarking
def test_planar_hand_progress() -> None:
    benchmark = planar_hand_setup_benchmark()
    benchmark.run_benchmark()
    # benchmark.print_results()
    # check_progress(benchmark)


# benchmark = floating_hand_setup_benchmark()
# benchmark.run_benchmark()
# benchmark.print_results()

# benchmark = kuka_setup_benchmark()
# benchmark.run_benchmark()
# benchmark.print_results()

# benchmark = planar_hand_setup_benchmark()
# benchmark.run_benchmark()
# benchmark.print_results()

# benchmark = spot_setup_benchmark()
# benchmark.run_benchmark()
# benchmark.print_results()
