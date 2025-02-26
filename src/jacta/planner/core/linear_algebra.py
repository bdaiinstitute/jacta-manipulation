# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
import numpy as np
import torch
from torch import FloatTensor, IntTensor


def transformation_matrix(
    rot: np.ndarray | None = None, pos: np.ndarray | None = None
) -> np.ndarray:
    """
    Returns a 4x4 transformation matrix given rotation matrix and translation vector.

    Parameters:
        rot (array-like): 3x3 rotation matrix.
        pos (array-like): Translation vector (3 elements).

    Returns:
        numpy.ndarray: 4x4 transformation matrix.
    """
    if rot is None:
        rot = np.eye(3)
    if pos is None:
        pos = np.zeros(3)
    # Create a 4x4 transformation matrix
    T = np.eye(4)
    # Set the rotation part
    T[:3, :3] = rot
    # Set the translation part
    T[:3, 3] = pos
    return T


def truncpareto_cdf(x: IntTensor, exponent: float, upper_bound: int) -> FloatTensor:
    distribution = (1 - x**-exponent) / (1 - 1 / upper_bound**exponent)
    torch.clamp(distribution, max=1, out=distribution)
    return distribution


def max_scaling(directions: FloatTensor, action_range: FloatTensor) -> FloatTensor:
    norms = torch.norm(directions, dim=-1)
    min_values = torch.min(action_range / torch.abs(directions), dim=-1).values
    return torch.where(norms != 0.0, min_values, torch.tensor(0.0))


def normalize(direction: FloatTensor) -> FloatTensor:
    if (norm := torch.norm(direction)) == 0:
        return direction
    else:
        return direction / norm


def normalize_multiple(directions: FloatTensor) -> FloatTensor:
    norms = torch.norm(directions, dim=-1, keepdim=True)
    return torch.where(norms != 0, directions / norms, directions)


def project_v_on_u(v: FloatTensor, u: FloatTensor) -> FloatTensor:
    """Calculates the projection of v on u.

    Raises:
        A ValueError if v and u are not vectors
    """
    if not len(v.shape) == 1:
        raise ValueError("v is not a vector!")
    if not len(u.shape) == 1:
        raise ValueError("u is not a vector!")
    return (torch.dot(v, u) / torch.norm(u) ** 2) * u


def gram_schmidt(basis_vectors: FloatTensor) -> FloatTensor:
    """Returns an orthonormal basis spanning the same dimension as the linearly independent basis vectors.

    This method assumes the basis vectors are linearly independent. If they aren't, you're going to have a bad time

    Args:
        basis_vectors: set of vectors that span a space. The columns are assumed to be individual vectors

    Raises:
        An ValueError if the basis vectors are not linearly independent
    """
    num_vectors, num_dims = basis_vectors.shape
    orthonormal_basis = torch.zeros((num_vectors, num_dims))
    for i in range(num_vectors):
        projections = torch.zeros(
            (num_dims),
        )
        basis_vector = basis_vectors[i, :]
        for j in range(i):
            projections += project_v_on_u(basis_vector, orthonormal_basis[j, :])
        orthonormal_basis[i, :] = normalize(basis_vector - projections)
        if torch.allclose(
            orthonormal_basis[i, :], torch.zeros(orthonormal_basis.shape[1])
        ):
            raise ValueError(
                f"Row {i} not linearly independent with other basis vectors"
            )
    return orthonormal_basis


def project_vectors_on_eigenspace(
    vectors: FloatTensor, orthonormal_basis: FloatTensor
) -> FloatTensor:
    """Given an eigenspace, projects the vector on the space.

    Args:
        vectors: (k, n) vector
        orthonormal: (m, n) sized orthonormal basis

    Returns:
        (k, n) vectors projected on the orthonormal basis
    """
    return (vectors @ orthonormal_basis.T) @ orthonormal_basis


# einsum wrappers for code clarity and to enforce einsum testing
# for einsum explanation, see https://ajcr.net/Basic-guide-to-einsum/

# vector-vector (dot) product


def einsum_ij_ij_i(A: FloatTensor, B: FloatTensor) -> FloatTensor:
    # i (j,) vectors, i (j,) vectors -> i scalars
    # transpose the first vector of each of the i vector-vector pairs and then multiply them
    return torch.einsum("ij,ij->i", A, B)


# matrix-vector product


def einsum_ij_kj_ki(A: FloatTensor, B: FloatTensor) -> FloatTensor:
    # 1 (i,j) matrix, k (j,) vectors -> k (i,) vectors
    # multiply the matrix with each of the k vectors
    return torch.einsum("ij,kj->ki", A, B)


def einsum_ijk_ik_ij(A: FloatTensor, B: FloatTensor) -> FloatTensor:
    # i (j,k) matrices, i (k,) vectors -> i (j,) vectors
    # multiply each of the i matrix-vector pairs
    return torch.einsum("ijk,ik->ij", A, B)


def einsum_ikj_ik_ij(A: FloatTensor, B: FloatTensor) -> FloatTensor:
    # i (k,j) matrices, i (k,) vectors -> i (j,) vectors
    # transpose the matrix of each matrix-vector pair and then multiply them
    return torch.einsum("ikj,ik->ij", A, B)


# matrix-matrix product


def einsum_jk_ikl_ijl(A: FloatTensor, B: FloatTensor) -> FloatTensor:
    # 1 (j,k) matrices, i (k,l) matrices -> i (j,l) matrices
    # multiply the matrix with each of the i matrices
    return torch.einsum("jk,ikl->ijl", A, B)


def einsum_ijk_ikl_ijl(A: FloatTensor, B: FloatTensor) -> FloatTensor:
    # i (j,k) matrices, i (k,l) matrices -> i (j,l) matrices
    # multiply each of the i matrix-matrix pairs
    return torch.einsum("ijk,ikl->ijl", A, B)


def einsum_ikj_ikl_ijl(A: FloatTensor, B: FloatTensor) -> FloatTensor:
    # i (k,j) matrices, i (k,l) matrices -> i (j,l) matrices
    # transpose the first matrix of each of the i matrix-matrix pairs and then multiply them
    return torch.einsum("ikj,ikl->ijl", A, B)


def einsum_ijk_ilk_ijl(A: FloatTensor, B: FloatTensor) -> FloatTensor:
    # i (j,k) matrices, i (l,k) matrices -> i (j,l) matrices
    # transpose the second matrix of each of the i matrix-matrix pairs and then multiply them
    return torch.einsum("ijk,ilk->ijl", A, B)
