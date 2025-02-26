# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
import torch

from jacta.planner.core.linear_algebra import (
    einsum_ij_ij_i,
    einsum_ij_kj_ki,
    einsum_ijk_ik_ij,
    einsum_ijk_ikl_ijl,
    einsum_ijk_ilk_ijl,
    einsum_ikj_ik_ij,
    einsum_ikj_ikl_ijl,
    einsum_jk_ikl_ijl,
)

torch.manual_seed(0)
atol = 1e-5  # For some reason when using torch instead of numpy it requires looser atol

dim_i = 16
dim_j = 8
dim_k = 4
dim_l = 2

# vector-vector (dot) product


def test_ij_ij_i() -> None:
    for _ in range(10):
        A = torch.randn(dim_i, dim_j)
        B = torch.randn(dim_i, dim_j)
        res0 = einsum_ij_ij_i(A, B)
        res1 = torch.stack([torch.matmul(A[ind], B[ind]) for ind in range(dim_i)])
        assert torch.allclose(res0, res1, atol=atol)


# matrix-vector product


def test_ij_kj_ki() -> None:
    for _ in range(1):
        A = torch.randn(dim_i, dim_j)
        B = torch.randn(dim_k, dim_j)
        res0 = einsum_ij_kj_ki(A, B)
        res1 = torch.stack([torch.matmul(A, B[ind]) for ind in range(dim_k)])
        assert torch.allclose(res0, res1, atol=atol)


def test_ijk_ik_ij() -> None:
    for _ in range(10):
        A = torch.randn(dim_i, dim_j, dim_k)
        B = torch.randn(dim_i, dim_k)
        res0 = einsum_ijk_ik_ij(A, B)
        res1 = torch.stack([torch.matmul(A[ind], B[ind]) for ind in range(dim_i)])
        assert torch.allclose(res0, res1, atol=atol)


def test_ikj_ik_ij() -> None:
    for _ in range(10):
        A = torch.randn(dim_i, dim_k, dim_j)
        B = torch.randn(dim_i, dim_k)
        res0 = einsum_ikj_ik_ij(A, B)
        res1 = torch.stack([torch.matmul(A[ind].mT, B[ind]) for ind in range(dim_i)])
        assert torch.allclose(res0, res1, atol=atol)


# matrix-matrix product


def test_jk_ikl_ijl() -> None:
    for _ in range(10):
        A = torch.randn(dim_j, dim_k)
        B = torch.randn(dim_i, dim_k, dim_l)
        res0 = einsum_jk_ikl_ijl(A, B)
        res1 = torch.stack([torch.matmul(A, B[ind]) for ind in range(dim_i)])
        assert torch.allclose(res0, res1, atol=atol)


def test_ijk_ikl_ijl() -> None:
    for _ in range(10):
        A = torch.randn(dim_i, dim_j, dim_k)
        B = torch.randn(dim_i, dim_k, dim_l)
        res0 = einsum_ijk_ikl_ijl(A, B)
        res1 = torch.stack([torch.matmul(A[ind], B[ind]) for ind in range(dim_i)])
        assert torch.allclose(res0, res1, atol=atol)


def test_ikj_ikl_ijl() -> None:
    for _ in range(10):
        A = torch.randn(dim_i, dim_k, dim_j)
        B = torch.randn(dim_i, dim_k, dim_l)
        res0 = einsum_ikj_ikl_ijl(A, B)
        res1 = torch.stack([torch.matmul(A[ind].mT, B[ind]) for ind in range(dim_i)])
        assert torch.allclose(res0, res1, atol=atol)


def test_ijk_ilk_ijl() -> None:
    for _ in range(10):
        A = torch.randn(dim_i, dim_j, dim_k)
        B = torch.randn(dim_i, dim_l, dim_k)
        res0 = einsum_ijk_ilk_ijl(A, B)
        res1 = torch.stack([torch.matmul(A[ind], B[ind].mT) for ind in range(dim_i)])
        assert torch.allclose(res0, res1, atol=atol)
