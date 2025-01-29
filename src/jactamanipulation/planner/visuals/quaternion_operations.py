# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
from __future__ import annotations

import meshcat.transformations as tf
import numpy as np
import numpy.typing as npt

_skew_symmetric_matrix = np.array(
    [[[0, 0, 0], [0, 0, 1], [0, -1, 0]], [[0, 0, -1], [0, 0, 0], [1, 0, 0]], [[0, 1, 0], [-1, 0, 0], [0, 0, 0]]]
)


def quaternion_inverse(quaternion: np.ndarray) -> np.ndarray:
    """Calculate the inverse of a quaternion.

    Args:
        quaternion (np.ndarray): Input quaternion [w, x, y, z].

    Result:
        np.ndarray: Inverse quaternion [w, -x, -y, -z].
    """
    inverse = np.zeros(4)
    inverse[0] = quaternion[0]
    inverse[1:4] = -quaternion[1:4]
    return inverse


def quaternion_to_transformation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a quaternion [w, x, y, z] into a 4x4 transformation matrix.

    Args:
        quaternion (np.ndarray): Quaternion represented as a numpy array [w, x, y, z].

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix(quaternion)
    return transform_matrix


def pose_to_transformation_matrix(pos: np.ndarray | None = None, quat: np.ndarray | None = None) -> np.ndarray:
    """Compute a 4x4 transformation matrix from position and quaternion.

    Args:
        pos (np.ndarray, optional): Position array [x, y, z]. Defaults to None.
        quat (np.ndarray, optional): Quaternion array [w, x, y, z]. Defaults to None.

    Returns:
        np.ndarray: Transformation matrix combining translation and rotation.
    """
    if pos is None:
        pos = np.zeros(3)
    if quat is None:
        quat = np.array([1, 0, 0, 0.0])
    transform = tf.translation_matrix(pos) @ quaternion_to_transformation_matrix(quat)
    return transform


def skew_symmetric(vector: npt.ArrayLike) -> np.ndarray:
    """Given a vector in R^3 construct a 3x3 skew-symmetric matrix.

    - If vector is 1-D array, then the result is a 3x3 matrix.


    - If vector is an Nx3 matrix, then the result is an Nx3x3 tensor representing a batch of skew-symmetric matrices.

    Args:
        vector: array_like a single or a batch of vectors.

    Result: ndarray of skew-symmetric matrices.

    Examples:
        >>> v = [1,2,3]
        >>> qo.skew_symmetric(v)
        array([[ 0, -3,  2],
               [ 3,  0, -1],
               [-2,  1,  0]])

        >>> v = [[1,2,3],[4,5,6]]
        >>> qo.skew_symmetric(v)
        array([[[ 0, -3,  2],
                [ 3,  0, -1],
                [-2,  1,  0]],

               [[ 0, -6,  5],
                [ 6,  0, -4],
                [-5,  4,  0]]])
    """
    return np.dot(vector, _skew_symmetric_matrix)


_left_action_matrix = np.array(
    [
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]],
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]],
        [[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]],
        [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]],
    ]
)


def left_action(quaternion: npt.ArrayLike) -> np.ndarray:
    """Construct a left action matrices L(q) from a quaternion or a list of quaternions.

    Left action matrix L(q) is defined as a linear operator for quaternion multiplication q_2 * q_1 = L(q_2) q_1.

    - If quaternion is 1-D array, then the result is a 4x4 matrix.


    - If quaternion is an Nx4 matrix, then the result is an Nx4x4 tensor representing a batch of left action matrices.

    Args:
        quaternion: array_like a single or a batch of quaternions.

    Result: ndarray of left action matrices.

    Examples:
        >>> q1 = [1,0,1,0]
        >>> q2 = [1,1,0,1]
        >>> qo.left_action(q2).dot(q1)
        array([1, 0, 1, 2])
    """
    return np.dot(quaternion, _left_action_matrix)


_right_action_matrix = np.array(
    [
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]],
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]],
        [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, -1, 0, 0]],
        [[0, 0, 0, 1], [0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
    ]
)


def right_action(quaternion: npt.ArrayLike) -> np.ndarray:
    """Construct a right action matrices R(q) from a quaternion or a list of quaternions.

    Right action matrix R(q) is defined as a linear operator for quaternion multiplication q_2 * q_1 = R(q_1) q_2.

    - If quaternion is 1-D array, then the result is a 4x4 matrix.


    - If quaternion is an Nx4 matrix, then the result is an Nx4x4 tensor representing a batch of right action matrices.

    Args:
        quaternion: array_like a single or a batch of quaternions.

    Result: ndarray of right action matrices.

    Examples:
        >>> q1 = [1,0,1,0]
        >>> q2 = [1,1,0,1]
        >>> qo.right_action(q1).dot(q2)
        array([1, 0, 1, 2])
    """
    return np.dot(quaternion, _right_action_matrix)


def action_inverse(action: npt.ArrayLike) -> np.ndarray:
    """Construct an inverse left or right action matrix assuming that it is orthogonal

    Construct an inverse left or right action matrix assuming that it is orthogonal, that is,
    it is constructed using unit quaternion.

    We assume action matrix orthogonality for computational efficiency.

    - If action is 2-D array, then the result is a transpose of this action.


    - If action is an Nx4x4 tensor, then the result is an Nx4x4 tensor representing a batch of inverse actions.

    Args:
        action: array_like a single or a batch of actions.

    Result: ndarray of inverse actions.
    """
    action = np.asarray(action)
    return np.transpose(action, (0, 2, 1)) if len(action.shape) == 3 else action.T


_vector_projection_matrix = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])


def vector_projection(quaternion: npt.ArrayLike) -> np.ndarray:
    """Project vector on imaginary quaternion space. This function supports batch projections."""
    return np.dot(quaternion, _vector_projection_matrix)


def project_action(action: npt.ArrayLike) -> np.ndarray:
    """Project action onto vector input space. This function supports batch projections."""
    return np.matmul(action, _vector_projection_matrix)


def rotation_matrix(quaternion: npt.ArrayLike) -> np.ndarray:
    """Compute a rotation matrix for a given quaternion.

    For computational efficiency we assume that the input is a unit quaternion.

    - If quaternion is a 1-D array, then the result is the corresponding 3x3 rotation matrix.

    - If quaternion is a 2-D matrix of shape Nx4, then the result is a Nx3x3 tensor of the corresponding rotation
    matrices.

    Args:
        quaternion: array_like a single or a batch of quaternions.

    Result: ndarray of rotation matrices.
    """
    projected_right_action = project_action(action_inverse(right_action(quaternion)))
    projected_left_action = project_action(action_inverse(left_action(quaternion)))
    return np.matmul(action_inverse(projected_left_action), projected_right_action)


def quaternion_to_quaternion_map_jacobian(
    argument_quaternion: npt.ArrayLike, map_jacobian: npt.ArrayLike, result_quaternion: npt.ArrayLike
) -> np.ndarray:
    """Compute a Jacobian w.r.t. tangent space coordinates of a quaternion-to-quaternion map f(q)->q'

    Compute a Jacobian w.r.t. tangent space coordinates of a quaternion-to-quaternion
    map f(q) -> q', given the Jacobian of the map in canonical coordinates, that is, df/dq.

    This function supports batch operation, in which input quaternions are Nx4 matrices, and map Jacobian is a Nx4x4
    tensor.

    Args:
        argument_quaternion: array_like quaternion q
        map_jacobian: array_like map Jacobian in canonical coordinates
        result_quaternion: array_like quaternion q'

    Result: ndarray of map Jacobian in tangent space.
    """
    projected_actions = project_action(left_action(np.vstack((argument_quaternion, result_quaternion))))
    batch_size = projected_actions.shape[0] // 2
    argument_action = projected_actions[:batch_size, :, :]
    result_action = projected_actions[batch_size:, :, :]
    jacobian = np.matmul(action_inverse(result_action), np.matmul(map_jacobian, argument_action))
    return jacobian if len(jacobian.shape) == 3 and jacobian.shape[0] > 1 else jacobian.reshape((3, 3))
