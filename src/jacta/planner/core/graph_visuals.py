# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import List, Optional

import matplotlib as mpl
import numpy as np
import torch
from pydrake.geometry import Meshcat, Rgba
from pydrake.perception import BaseField, Fields, PointCloud
from torch import FloatTensor

from jacta.planner.core.graph import Graph
from jacta.planner.core.logger import Logger


def rgba_palette(index: int, transparency: float = 1.0) -> Rgba:
    if index % 8 == 0:
        rgb = [255, 255, 0]
    elif index % 8 == 1:
        rgb = [175, 238, 30]
    elif index % 8 == 2:
        rgb = [255, 165, 0]
    elif index % 8 == 3:
        rgb = [199, 21, 133]
    elif index % 8 == 4:
        rgb = [65, 105, 225]
    elif index % 8 == 5:
        rgb = [218, 112, 214]
    elif index % 8 == 6:
        rgb = [250, 128, 114]
    elif index % 8 == 7:
        rgb = [50, 50, 50]
    return Rgba(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, transparency)


def color_gradient(color: Rgba, steps: int) -> FloatTensor:
    if steps == 1:
        colors = [color]
    else:
        colors = []
        for i in range(steps):
            rgb = color.rgba[0:3] * (
                1 - i / ((steps - 1) * 2)
            )  # full color to "half" color
            colors.append(Rgba(*rgb, color.rgba[3]))
    return colors


def display_point_cloud(
    meshcat_vis: Meshcat,
    points: np.ndarray | torch.Tensor,
    path: str = "/points",
    point_size: float = 0.01,
    color: Rgba = None,
) -> None:
    if isinstance(points, torch.Tensor):
        points = np.array([points.cpu().numpy()])
    num_points = np.shape(points)[0]
    if num_points == 0:
        return
    assert np.shape(points)[1] == 3

    if color is None:
        color = Rgba(0.2, 0.1, 0.2, 1.0)

    fields = Fields(BaseField.kXYZs)
    point_cloud = PointCloud(num_points, fields=fields)
    point_cloud.mutable_xyzs()[:] = points.T
    meshcat_vis.SetObject(
        path,
        point_cloud,
        point_size=point_size,
        rgba=color,
    )


def display_segments(
    meshcat_vis: Meshcat,
    start: np.ndarray | torch.Tensor,
    end: np.ndarray | torch.Tensor,
    path: str = "/segments",
    line_width: float = 0.01,
    color: Rgba = None,
) -> None:
    if isinstance(start, torch.Tensor):
        start = start.cpu().numpy()
    if isinstance(end, torch.Tensor):
        end = end.cpu().numpy()
    num_points = np.shape(start)[0]
    if num_points == 0:
        return
    assert np.shape(start)[1] == 3
    assert np.shape(end)[1] == 3
    assert np.shape(end)[0] == num_points

    if color is None:
        color = Rgba(0.2, 0.1, 0.2, 1.0)

    meshcat_vis.SetLineSegments(
        path,
        start.T,
        end.T,
        line_width=line_width,
        rgba=color,
    )


def display_colormap_point_cloud(
    meshcat_vis: Meshcat,
    points: np.ndarray | torch.Tensor,
    rewards: np.ndarray | torch.Tensor,
    path: str = "/colormap_points",
    point_size: float = 0.01,
    num_color_bins: int = 12,
    transparency: float = 0.7,
) -> None:
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(rewards, torch.Tensor):
        rewards = rewards.cpu().numpy()
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

    # sort points into bins
    point_bins = [np.zeros((1, 3)) for i in range(num_color_bins)]
    for i in range(num_points):
        point = points[i : i + 1, :]
        reward = scaled_rewards[i]
        id_bin = np.floor(reward * num_color_bins)
        id_bin = int(min(id_bin, num_color_bins - 1))
        point_bins[id_bin] = np.append(point_bins[id_bin], point, axis=0)

    # display points bin by bin
    viridis = mpl.colormaps["viridis"]
    for i in range(num_color_bins):
        display_point_cloud(
            meshcat_vis,
            point_bins[i],
            path=path + "/bin_" + str(i),
            point_size=point_size,
            color=Rgba(*viridis(i / num_color_bins, transparency)),
        )


def display_edges_by_category(
    meshcat_vis: Meshcat,
    starts: np.ndarray | torch.Tensor,
    ends: np.ndarray | torch.Tensor,
    categories: List,
    edge_size: int = 1,  # in pixels
    path: str = "/3d_graph/categories",
) -> None:
    num_points = np.shape(starts)[0]
    if num_points == 0:
        return
    assert np.shape(ends)[0] == num_points

    category_set = set(categories)
    for k, category in enumerate(category_set):
        indices = [
            i for i, c in enumerate(categories) if c == category and i < num_points
        ]
        display_segments(
            meshcat_vis,
            starts[indices],
            ends[indices],
            path=path + "/" + category,
            line_width=edge_size,
            color=rgba_palette(k),
        )


def display_3d_graph(
    graph: Graph,
    logger: Logger,
    meshcat_vis: Meshcat,
    vis_scale: Optional[FloatTensor] = None,
    vis_indices: Optional[List] = None,
    node_size: float = 0.01,
    start_goal_size: float = 0.08,
    edge_size: int = 1,  # in pixels
    best_path_edge_size: int = 4,  # in pixels
    segment_color: Optional[Rgba] = None,
    best_path_color: Optional[Rgba] = None,
    node_transparency: float = 0.7,
    display_segment: bool = True,
    display_best_path: bool = True,
    display_reward_colormap: bool = True,
    node_cap: Optional[int] = None,
    reset_visualizer: bool = True,
    search_index: int = 0,
) -> None:
    if vis_scale is None:
        vis_scale = torch.ones(3)
    if vis_indices is None:
        vis_indices = [0, 1, 2]
    if segment_color is None:
        segment_color = Rgba(1, 0.4, 0.2, 1)
    if best_path_color is None:
        best_path_color = Rgba(1, 0.4, 0.2, 1)

    assert len(vis_indices) == 3

    states = graph.states

    # reset visualizer
    if reset_visualizer:
        meshcat_vis.Delete("/3d_graph")

    # cap the number of nodes that will be displayed
    if node_cap is None:
        node_cap = graph.number_of_nodes()
    displayed_nodes = graph.get_active_main_ids(search_index=search_index)[:node_cap]

    # roots
    roots = vis_scale * states[graph.get_root_ids(), :][:, vis_indices]

    # goals
    goal = vis_scale * graph.goal_states[search_index, vis_indices]
    sub_goal = vis_scale * graph.sub_goal_states[search_index, vis_indices]
    goals = torch.vstack((goal, sub_goal))

    # nodes
    nodes = vis_scale * states[displayed_nodes, :][..., vis_indices]

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
    starts = 0.10 * ends + 0.90 * starts

    colors = color_gradient(Rgba(1, 1, 0, 0.7), len(roots))
    for i, root in enumerate(roots):
        display_point_cloud(
            meshcat_vis,
            root,
            path=f"/3d_graph/roots_{i}",
            point_size=start_goal_size,
            color=colors[i],
        )

    colors = color_gradient(Rgba(0.0, 0.0, 1.0, 0.7), len(goals))
    for i, goal in enumerate(goals):
        display_point_cloud(
            meshcat_vis,
            goal,
            path=f"/3d_graph/goals_{i}",
            point_size=start_goal_size,
            color=colors[i],
        )

    if display_segment:
        display_segments(
            meshcat_vis,
            starts,
            ends,
            path="/3d_graph/edges",
            line_width=edge_size,
            color=segment_color,
        )

    if display_reward_colormap:
        rewards = graph.rewards[displayed_nodes]
        display_colormap_point_cloud(
            meshcat_vis,
            nodes,
            rewards,
            path="/3d_graph/nodes",
            point_size=node_size,
            num_color_bins=12,
            transparency=node_transparency,
        )
    else:
        display_point_cloud(
            meshcat_vis,
            nodes,
            path="/3d_graph/nodes",
            point_size=node_size,
            color=Rgba(0.2, 0.2, 0.2, node_transparency),
        )

    if display_best_path:
        # find the node closest to the goal node from precomputed distances
        best_id = graph.get_best_id(
            reward_based=False, search_indices=torch.tensor([search_index])
        ).item()
        path_ids = graph.shortest_path_to(best_id)
        path = states[path_ids, :][:, vis_indices] * vis_scale
        starts = path[:-1, :]
        ends = path[1:, :]
        display_segments(
            meshcat_vis,
            starts,
            ends,
            path="/3d_graph/best_path/path",
            line_width=best_path_edge_size,
            color=best_path_color,
        )

        if logger is not None and logger.graph.params.num_parallel_searches == 1:
            selection_strategies, action_strategies = logger.simple_path_statistics()
            display_edges_by_category(
                meshcat_vis,
                starts,
                ends,
                action_strategies,
                edge_size=2 * best_path_edge_size,
                path="/3d_graph/best_path/action_strategies",
            )
            display_edges_by_category(
                meshcat_vis,
                starts,
                ends,
                selection_strategies,
                edge_size=2 * best_path_edge_size,
                path="/3d_graph/best_path/selection_strategies",
            )
