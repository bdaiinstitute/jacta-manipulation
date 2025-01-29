# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import numpy as np
import numpy.testing as np_test
from scipy.spatial.transform import Rotation

from dexterity.planner.visuals import quaternion_operations as qo

_scipy_to_quat = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])


def test_left_action() -> None:
    # single quaternion test
    np_test.assert_almost_equal(
        qo.left_action([1, 10, 100, 1000]),
        np.vstack(
            ([1, -10, -100, -1000], np.hstack(([[10], [100], [1000]], np.eye(3) + qo.skew_symmetric([10, 100, 1000]))))
        ),
    )
    # test batch quaternions
    left_actions = qo.left_action([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    np_test.assert_array_almost_equal(left_actions[0], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    np_test.assert_array_almost_equal(left_actions[1], [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])
    np_test.assert_array_almost_equal(left_actions[2], [[0, 0, -1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, -1, 0, 0]])
    np_test.assert_array_almost_equal(left_actions[3], [[0, 0, 0, -1], [0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])

    # test group action
    r1 = Rotation.random()
    r2 = Rotation.random()
    r3 = r2 * r1
    q1 = r1.as_quat().dot(_scipy_to_quat)
    q2 = r2.as_quat().dot(_scipy_to_quat)
    q3 = r3.as_quat().dot(_scipy_to_quat)
    np_test.assert_almost_equal(qo.left_action(q2).dot(q1), q3)


def test_right_action() -> None:
    # single quaternion test
    np_test.assert_almost_equal(
        qo.right_action([1, 10, 100, 1000]),
        np.vstack(
            ([1, -10, -100, -1000], np.hstack(([[10], [100], [1000]], np.eye(3) - qo.skew_symmetric([10, 100, 1000]))))
        ),
    )
    # test batch quaternions
    right_actions = qo.right_action([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    np_test.assert_array_almost_equal(right_actions[0], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    np_test.assert_array_almost_equal(right_actions[1], [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
    np_test.assert_array_almost_equal(right_actions[2], [[0, 0, -1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]])
    np_test.assert_array_almost_equal(right_actions[3], [[0, 0, 0, -1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]])

    # test group action
    r1 = Rotation.random()
    r2 = Rotation.random()
    r3 = r2 * r1
    q1 = r1.as_quat().dot(_scipy_to_quat)
    q2 = r2.as_quat().dot(_scipy_to_quat)
    q3 = r3.as_quat().dot(_scipy_to_quat)
    np_test.assert_almost_equal(qo.right_action(q1).dot(q2), q3)


def test_rotation_matrix() -> None:
    # single quaternion test
    rotation = Rotation.random()
    quaternion = rotation.as_quat().dot(_scipy_to_quat)
    rotation_matrix = qo.rotation_matrix(quaternion)
    expected_rotation_matrix = rotation.as_matrix()
    np_test.assert_almost_equal(rotation_matrix, expected_rotation_matrix)

    # test batch quaternions
    rotations = [Rotation.random() for _ in range(10)]
    quaternions = np.vstack([rot.as_quat().dot(_scipy_to_quat) for rot in rotations])
    rotation_matrices = qo.rotation_matrix(quaternions)
    for i, rot in enumerate(rotations):
        np_test.assert_almost_equal(rotation_matrices[i], rot.as_matrix())


def test_quaternion_to_quaternion_map_jacobian() -> None:
    quaternion = [1, 0, 0, 0]
    identity = np.eye(4)
    np_test.assert_almost_equal(qo.quaternion_to_quaternion_map_jacobian(quaternion, identity, quaternion), np.eye(3))
    np_test.assert_almost_equal(
        qo.quaternion_to_quaternion_map_jacobian(
            np.vstack((quaternion, quaternion)), np.array((identity, identity)), np.vstack((quaternion, quaternion))
        ),
        np.array((np.eye(3), np.eye(3))),
    )

    # TODO: we need some nontrivial tests
