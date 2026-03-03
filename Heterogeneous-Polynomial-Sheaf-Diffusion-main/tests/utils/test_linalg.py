import torch
import pytest
from polynsd.utils.linalg import householder_orgqr


@pytest.mark.parametrize("n", [4, 8])
def test_householder_orthogonality(n, device, dtype):
    # Use the fixtures directly as arguments
    params = torch.randn(n, n, device=device, dtype=dtype)
    eye = torch.eye(n, device=device, dtype=dtype)

    A = params.tril(diagonal=-1) + eye
    Q = householder_orgqr(A)

    # Check Q^T @ Q == I
    reconstruction = Q.T @ Q
    assert torch.allclose(reconstruction, eye, atol=1e-5)
