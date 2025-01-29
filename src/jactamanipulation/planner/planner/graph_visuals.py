# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import matplotlib as mpl
import meshcat.geometry as g
import numpy as np
import torch
from meshcat.visualizer import Visualizer
from torch import FloatTensor

from jactamanipulation.planner.planner.graph import Graph
from jactamanipulation.planner.planner.logger import Logger
from jactamanipulation.planner.visuals.meshcat_visualizer import rgb_float_to_hex


def rgba_palette(index: int, transparency: float = 1.0) -> list[float]:
    """Predefined set of colors for rendering."""
    if index % 8 == 0:
        rgb = [255, 255, 0]  # yellow
    elif index % 8 == 1:
        rgb = [175, 238, 30]  # light green
    elif index % 8 == 2:
        rgb = [255, 165, 0]  # orange
    elif index % 8 == 3:
        rgb = [199, 21, 133]  # purple
    elif index % 8 == 4:
        rgb = [65, 105, 225]  # blue
    elif index % 8 == 5:
        rgb = [218, 112, 214]  # magenta
    elif index % 8 == 6:
        rgb = [250, 128, 114]  # light red
    elif index % 8 == 7:
        rgb = [50, 50, 50]  # gray
    return [rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, transparency]


def color_gradient(rgba: list[float], steps: int) -> FloatTensor:
    if steps == 1:
        colors = [rgba]
    else:
        colors = []
        for i in range(steps):
            rgb = np.array(rgba[0:3]) * (1 - i / ((steps - 1) * 2))  # full color to "half" color
            colors.append(list(rgb) + rgba[3:4])
    return colors


def display_point_cloud(
    visualizer: Visualizer,
    points: np.ndarray | torch.Tensor,
    name: str = "points",
    point_size: float = 0.01,
    color: np.ndarray | list[float] | None = None,
) -> None:
    """Display point cloud in Meshcat with a specific RGB color and point size.

    Args:
        visualizer (Visualizer): Visualizer
        points (np.ndarray | torch.Tensor): [num_points, 3]
        name (str, optional): Point cloud name. Defaults to "points".
        point_size (float, optional): Point size. Defaults to 0.01.
        color (np.ndarray | list[float] | None, optional): Points color. Defaults to None.
    """
    if isinstance(points, torch.Tensor):
        points = np.array(points.cpu().numpy()).astype(np.float32)
    if points.ndim == 1:
        points = points[np.newaxis, :]
    num_points = np.shape(points)[0]
    if num_points == 0:
        return
    assert np.shape(points)[1] == 3

    if color is None:
        color = [0.2, 0.1, 0.2]

    if isinstance(color, list):
        hex = rgb_float_to_hex(color)
        material = g.PointsMaterial(size=point_size, color=hex)
        geometry = g.PointsGeometry(points.T)
    else:
        material = g.PointsMaterial(size=point_size)
        geometry = g.PointsGeometry(points.T, color.T)

    visualizer[name].set_object(g.Points(geometry, material))


def display_segments(
    visualizer: Visualizer,
    start: np.ndarray | torch.Tensor,
    end: np.ndarray | torch.Tensor,
    name: str = "segments",
    line_width: float = 1,  # in pixels
    color: list[float] | None = None,
) -> None:
    """Display segments in Meshcat with a specific RGB color and line width.

    Args:
        visualizer (Visualizer): Meshcat Visualizer
        start (np.ndarray | torch.Tensor): [num_points, 3]
        end (np.ndarray | torch.Tensor): [num_points, 3]
        name (str, optional): Segments name. Defaults to "segments".
        line_width (list[float], optional): Line width. Defaults to 1.
        color (list[float] | None, optional): Color. Defaults to None
    """
    if isinstance(start, torch.Tensor):
        start = start.cpu().numpy().astype(np.float32)
    if isinstance(end, torch.Tensor):
        end = end.cpu().numpy().astype(np.float32)
    num_points = np.shape(start)[0]
    if num_points == 0:
        return
    assert np.shape(start)[1] == 3
    assert np.shape(end)[1] == 3
    assert np.shape(end)[0] == num_points

    if color is None:
        color = [0.2, 0.1, 0.2]
    hex = rgb_float_to_hex(color)

    interleaved_array = np.empty((2 * num_points, 3))
    interleaved_array[0::2, :] = start
    interleaved_array[1::2, :] = end
    interleaved_array = interleaved_array.astype(np.float32)

    geometry = g.PointsGeometry(interleaved_array.T)
    material = g.LineBasicMaterial(linewidth=line_width, color=hex)
    visualizer[name].set_object(g.LineSegments(geometry, material))


def display_colormap_point_cloud(
    visualizer: Visualizer,
    points: np.ndarray | torch.Tensor,
    rewards: np.ndarray | torch.Tensor,
    is_terminal: np.ndarray | torch.Tensor,
    name: str = "colormap_points",
    point_size: float = 0.01,
    num_color_bins: int = 12,
) -> None:
    """Display point cloud in Meshcat with a specific RGB color dependent on individual point reward.

    Args:
        visualizer (Visualizer): Meshcat Visualizer
        points (np.ndarray | torch.Tensor): [num_points, 3]
        rewards (np.ndarray | torch.Tensor): [num_points]
        is_terminal (np.ndarray | torch.Tensor): is_terminal
        name (str, optional): Colormap name. Defaults to "colormap_points".
        point_size (float, optional): Point size. Defaults to 0.01.
        num_color_bins (int, optional): Number of color bins. Defaults to 12.
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(rewards, torch.Tensor):
        rewards = rewards.cpu().numpy()
    if isinstance(is_terminal, torch.Tensor):
        is_terminal = is_terminal.cpu().numpy()
    num_points = np.shape(points)[0]
    if num_points == 0:
        return
    assert np.shape(points)[1] == 3
    assert len(rewards) == num_points

    # scale rewards between 0 and 1
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    if min_reward != max_reward:
        scaled_rewards = (rewards - min_reward) / (max_reward - min_reward)
    else:
        scaled_rewards = rewards - min_reward

    # display points bin by bin
    viridis = mpl.colormaps["viridis"]
    colors = viridis(scaled_rewards)[:, 0:3]

    # add terminal color (red)
    colors[is_terminal, :] = [1, 0, 0]

    display_point_cloud(visualizer, points, name=name, point_size=point_size, color=colors)


def display_segments_by_category(
    visualizer: Visualizer,
    starts: np.ndarray | torch.Tensor,
    ends: np.ndarray | torch.Tensor,
    categories: list,
    line_width: int = 1,  # in pixels
    name: str = "categories",
) -> None:
    """Display segments in Meshcat with a specific RGB color dependent on individual point category.

    Args:
        visualizer (Visualizer): Meshcat Visualizer
        starts (np.ndarray | torch.Tensor): [num_points, 3]
        ends (np.ndarray | torch.Tensor): [num_points, 3]
        categories (list): [num_points] list of strings
        line_width (int, optional): Line width. Defaults to 1.
        name (str, optional): Segments name. Defaults to 'categories'
    """
    num_points = np.shape(starts)[0]
    if num_points == 0:
        return
    assert np.shape(ends)[0] == num_points

    category_set = set(categories)
    for k, category in enumerate(category_set):
        indices = [i for i, c in enumerate(categories) if c == category and i < num_points]
        if len(category) == 0:
            category = "undefined_category"
        display_segments(
            visualizer,
            starts[indices],
            ends[indices],
            name=name + "/" + category,
            line_width=line_width,
            color=rgba_palette(k)[0:3],
        )


def display_3d_graph(
    graph: Graph,
    logger: Logger,
    visualizer: Visualizer,
    vis_scale: FloatTensor | None = None,
    vis_indices: list | None = None,
    node_size: float = 0.01,
    start_goal_size: float = 0.06,
    edge_size: int = 1,  # in pixels
    best_path_edge_size: int = 4,  # in pixels
    segment_color: list[float] | None = None,
    best_path_color: list[float] | None = None,
    node_transparency: float = 0.7,
    display_segment: bool = True,
    display_best_path: bool = True,
    display_reward_colormap: bool = True,
    reset_visualizer: bool = True,
    search_index: int = 0,
) -> None:
    """Display search graph in Meshcat, each node is an vertex, each edge in an action linking 2 vertices.

    Args:
        graph (Graph): Planner Graph
        logger (Logger): Planner Logger
        visualizer (Visualizer): Meshcat Visualizer
        vis_scale (FloatTensor | None, optional): Scale. Defaults to None.
        vis_indices (list | None, optional): Indices. Defaults to None.
        node_size (float, optional): Node size. Defaults to 0.01.
        start_goal_size (float, optional): Start goal size. Defaults to 0.06.
        edge_size (int, optional): Edge size. Defaults to 1.
        best_path_edge_size (int, optional): Best path edge size. In pixels. Defaults to 4.
        segment_color (list[float] | None, optional): Segment color. Defaults to None
        best_path_color (list[float] | None, optional): Best path color. Defaults to None.
        node_transparency (float, optional): Node transparency. Defaults to 0.7.
        display_segment (bool, optional): Whether to display segment or not. Defaults to True.
        display_best_path (bool, optional): Whether to display best path or not. Defaults to True.
        display_reward_colormap (bool, optional): Whether to display reward colormap or not. Defaults to True.
        reset_visualizer (bool, optional): Whether to reset visualizer or not. Defaults to True.
        search_index (int, optional): Search index. Defaults to 0.
    """
    if vis_scale is None:
        vis_scale = torch.ones(3)
    if vis_indices is None:
        vis_indices = [0, 1, 2]
    if segment_color is None:
        segment_color = [0.9, 0.9, 1.0]
    if best_path_color is None:
        best_path_color = [1, 0.4, 0.2]

    assert len(vis_indices) == 3

    states = graph.states

    # reset visualizer
    if reset_visualizer:
        visualizer["3d_graph"].delete()

    # find all active nodes (including main and sub nodes)
    valid_ids = graph.ids[graph.active_ids]
    search_indices = graph.node_id_to_search_index_map[valid_ids]
    displayed_nodes = valid_ids[search_indices == search_index]

    # find all active main nodes
    displayed_main_nodes = graph.get_active_main_ids(search_index=search_index)

    # roots
    roots = vis_scale * states[graph.get_root_ids(), :][:, vis_indices]

    # goals
    goal = vis_scale * graph.goal_states[search_index, vis_indices]
    sub_goal = vis_scale * graph.sub_goal_states[search_index, vis_indices]
    goals = torch.vstack((goal, sub_goal))
    assert goals.shape[0] == 2

    # nodes
    nodes = vis_scale * states[displayed_nodes, :][..., vis_indices]
    main_nodes = vis_scale * states[displayed_main_nodes, :][..., vis_indices]

    # segments
    parents = []
    children = []
    for idx in displayed_nodes:
        parent_id = graph.parents[idx]
        if parent_id in displayed_nodes and parent_id != idx:
            parents.append(parent_id)
            children.append(idx)

    starts = states[parents, :][:, vis_indices] * vis_scale
    ends = states[children, :][:, vis_indices] * vis_scale
    starts = 0.05 * ends + 0.95 * starts

    colors = color_gradient([1, 1, 0, 0.7], len(roots))
    for i, root in enumerate(roots):
        display_point_cloud(
            visualizer,
            root,
            name=f"3d_graph/roots_{i}",
            point_size=start_goal_size,
            color=colors[i][0:3],
        )

    colors = color_gradient([0, 0, 1, 0.7], len(goals))
    for i, goal in enumerate(goals):
        if i == 0:
            name = "3d_graph/goals/goal"
        else:
            name = "3d_graph/goals/sub_goal"
        display_point_cloud(
            visualizer,
            goal,
            name=name,
            point_size=start_goal_size,
            color=colors[i][0:3],
        )

    if display_segment:
        display_segments(
            visualizer,
            starts,
            ends,
            name="3d_graph/edges",
            line_width=edge_size,
            color=segment_color,
        )

    display_point_cloud(
        visualizer,
        nodes,
        name="3d_graph/nodes/all_nodes",
        point_size=node_size / 2.5,
        color=[0.2, 0.2, 0.2],
    )
    if display_reward_colormap:
        rewards = graph.rewards[displayed_main_nodes]
        is_terminal = graph.terminal[displayed_main_nodes]
        display_colormap_point_cloud(
            visualizer,
            main_nodes,
            rewards,
            is_terminal,
            name="3d_graph/nodes/main_nodes",
            point_size=node_size,
        )
    else:
        display_point_cloud(
            visualizer,
            main_nodes,
            name="3d_graph/nodes/main_nodes",
            point_size=node_size,
            color=[0.2, 0.2, 0.2],
        )

    if display_best_path:
        # find the node closest to the goal node from precomputed distances
        best_id = graph.get_best_id(reward_based=False, search_indices=torch.tensor([search_index])).item()
        path_ids = graph.shortest_path_to(best_id)
        path = states[path_ids, :][:, vis_indices] * vis_scale
        starts = path[:-1, :]
        ends = path[1:, :]
        display_segments(
            visualizer,
            starts,
            ends,
            line_width=best_path_edge_size,
            name="3d_graph/best_path/path",
            color=best_path_color,
        )

        if logger is not None and logger.graph.params.num_parallel_searches == 1:  # TODO: fix for parallel searches
            # TODO: this must be wrong as we don't pass the starts and ends to to get the statistics
            selection_strategies, action_strategies = logger.simple_path_statistics()
            display_segments_by_category(
                visualizer,
                starts,
                ends,
                action_strategies,
                line_width=2 * best_path_edge_size,
                name="3d_graph/best_path/action_strategies",
            )
            display_segments_by_category(
                visualizer,
                starts,
                ends,
                selection_strategies,
                line_width=2 * best_path_edge_size,
                name="3d_graph/best_path/selection_strategies",
            )
