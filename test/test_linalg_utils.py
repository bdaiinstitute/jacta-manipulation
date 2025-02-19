# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
# %%
from math import sqrt

import pytest
import torch
from scipy.stats import truncpareto
from torch import tensor

from jacta.planner.core.clipping_method import box_scaling
from jacta.planner.core.linear_algebra import (
    gram_schmidt,
    normalize,
    project_v_on_u,
    project_vectors_on_eigenspace,
    truncpareto_cdf,
)

torch.manual_seed(0)


def check_orthogonal(A: torch.FloatTensor) -> bool:
    ATA = torch.matmul(torch.transpose(A, 0, 1), A)
    Id = torch.eye(A.shape[0])
    return torch.allclose(ATA, Id, atol=1e-4, rtol=1e-4)


def test_project_v_on_u() -> None:
    v = tensor([1, 0, 0, 0.0])
    u = tensor([0, 1, 0, 0.0])
    p_v_u = project_v_on_u(v, u)
    assert torch.allclose(p_v_u, torch.zeros((4,)))

    p_v_u = project_v_on_u(v, v)
    assert torch.allclose(p_v_u, v)

    v = tensor([1, 5, 0, 0.0])
    p_v_u = project_v_on_u(v, u)
    assert torch.allclose(p_v_u, tensor([0, 5, 0, 0.0]))

    v = tensor([2.7, 0, 0, 3.4])
    u = tensor([1, 1, 1, 1.0])
    p_v_u = project_v_on_u(v, u)
    assert torch.allclose(p_v_u, torch.ones((4,)) * torch.sum(v) / 4)


def test_gram_schmidt() -> None:
    space = tensor([[1, 0, 0], [0, 6, 0], [0, 0, 10.0]])
    basis = gram_schmidt(space)
    assert torch.allclose(basis, torch.eye(3))
    assert check_orthogonal(basis)

    space = tensor([[0, 1], [1, 1.0]])
    basis = gram_schmidt(space)
    assert torch.allclose(basis, tensor([[0, 1], [1, 0.0]]))
    assert check_orthogonal(basis)

    space = tensor([[1, -1, 1], [1, 0, 1], [1, 1, 2.0]])
    basis = gram_schmidt(space)
    col_one = normalize(tensor([1, -1, 1.0]))
    col_two = normalize(tensor([1 / 3, 2 / 3, 1 / 3]))
    col_three = normalize(tensor([-1 / 2, 0, 1 / 2]))
    actual_basis = torch.vstack((col_one, col_two, col_three))
    assert check_orthogonal(actual_basis)
    assert check_orthogonal(basis)
    assert torch.allclose(basis, actual_basis, atol=1e-4)

    space = tensor([[0, 1], [0, 1.0]])
    with pytest.raises(ValueError):
        basis = gram_schmidt(space)


def test_project_vectors_on_eigenspace() -> None:
    iden_space = torch.eye(3)
    action = torch.rand((1, 3))
    assert torch.allclose(
        project_vectors_on_eigenspace(action, iden_space), action, atol=1e-6
    )

    space = tensor([[0, 1], [1, 1.0]])
    basis = gram_schmidt(space)
    action = torch.rand((1, 2))
    assert torch.allclose(
        project_vectors_on_eigenspace(action, basis), action, atol=1e-6
    )

    basis = tensor([[1 / sqrt(2), 1 / sqrt(2), 0, 0], [0, 0, 1, 0]])
    action = tensor([[sqrt(2), 0, 1, 1]])
    real_results = tensor([[1 / sqrt(2), 1 / sqrt(2), 1, 0]])
    assert torch.allclose(
        project_vectors_on_eigenspace(action, basis), real_results, atol=1e-4
    )

    basis = tensor([[1, 0, 0], [0, 1, 0.0]])
    action = tensor([[1, 0, 1], [1, 1, 0.0]])
    real_results = tensor([[1, 0, 0], [1, 1, 0.0]])
    assert torch.allclose(
        project_vectors_on_eigenspace(action, basis), real_results, atol=1e-4
    )


def test_box_scaling() -> None:
    v = tensor([10, 20.0])
    v_min = -tensor([1, 1.0])
    v_max = +tensor([1, 1.0])
    v_bar = box_scaling(v, v_min, v_max)
    assert torch.allclose(v_bar, tensor([0.5, 1]))

    v = tensor([-10, -20.0])
    v_min = -tensor([1, 1.0])
    v_max = +tensor([1, 1.0])
    v_bar = box_scaling(v, v_min, v_max)
    assert torch.allclose(v_bar, tensor([-0.5, -1]))

    v = tensor([-10, 20.0])
    v_min = -tensor([1, 1.0])
    v_max = +tensor([1, 1.0])
    v_bar = box_scaling(v, v_min, v_max)
    assert torch.allclose(v_bar, tensor([-0.5, 1]))

    v = tensor([10, 20.0])
    v_min = -tensor([0, 0.0])
    v_max = +tensor([10, 10.0])
    v_bar = box_scaling(v, v_min, v_max)
    assert torch.allclose(v_bar, tensor([5 + 5 / 3, 10]))

    v = tensor([-10, -20.0])
    v_min = -tensor([0, 0.0])
    v_max = +tensor([10, 10.0])
    v_bar = box_scaling(v, v_min, v_max)
    assert torch.allclose(v_bar, tensor([2, 0.0]))

    v = tensor([-10, 20.0])
    v_min = -tensor([0, 0.0])
    v_max = +tensor([10, 10.0])
    v_bar = box_scaling(v, v_min, v_max)
    assert torch.allclose(v_bar, tensor([0, 10.0]))


def test_truncpareto_cdf() -> None:
    for length in [1, 10, 100]:
        for exponent in [0.1, 1.0, 10.0]:
            upper_bound = length + 1
            x = torch.arange(1, upper_bound)
            scipy_pareto = tensor(truncpareto.cdf(x.cpu(), exponent, upper_bound))
            jacta_pareto = tensor(
                truncpareto_cdf(x, exponent, upper_bound), dtype=torch.float64
            )
            assert torch.allclose(scipy_pareto, jacta_pareto)
