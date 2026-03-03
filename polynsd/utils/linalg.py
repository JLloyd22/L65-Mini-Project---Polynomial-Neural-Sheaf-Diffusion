import torch
import torch_sparse


def householder_orgqr(A: torch.Tensor) -> torch.Tensor:
    """Native PyTorch replacement for torch_householder_orgqr.

    Converts a matrix of Householder reflectors (lower triangular)
    into an orthogonal matrix Q.

    Args:
        A: Tensor of shape (..., M, N) where columns are reflectors.

    Returns:
        Orthogonal matrix Q of shape (..., M, M).
    """
    # Calculate tau: 2 / ||v||^2 for each column
    # dim=-2 represents the 'M' dimension (the height of the reflectors)
    tau = 2.0 / (A**2).sum(dim=-2)

    # torch.linalg.householder_product is the native equivalent to ORGQR
    return torch.linalg.householder_product(A, tau)

def cayley_transform(A: torch.Tensor) -> torch.Tensor:
    """Cayley transform: maps skew-symmetric matrices to orthogonal matrices. O = (I
    - A)(I + A)^{-1} where A is skew-symmetric. Misses SO(d) matrices with eigenvalue
    -1. 180-degree rotations in even dimensions are not representable. (not an issue).
    """
    d = A.shape[-1]
    I = torch.eye(d, device=A.device, dtype=A.dtype)  # noqa: E741
    # Ensure A is skew-symmetric: A_skew = (A - A^T) / 2
    A_skew = (A - A.transpose(-2, -1)) / 2  # Projects d^2 to d(d-1)/2
    return torch.linalg.solve(I + A_skew, I - A_skew)

def estimate_largest_eig(index_pair, vals, N, num_iter=10):
    """Approximate the largest eigenvalue λ_max of the sparse Laplacian defined by (index_pair, vals)
    via power iteration.
    - index_pair: a tuple (idx_i, idx_j) of 1-D long tensors of length nnz
    - vals:        a 1-D tensor of length nnz
    - N:           the dimension of the square matrix (i.e. number of rows/cols).
    """
    if isinstance(index_pair, tuple):  # (row_idx, col_idx)
        index_pair = torch.stack(index_pair, dim=0)
    # initialize with a random vector of shape (N,1)
    x = torch.randn((N, 1), device=vals.device)
    for _ in range(num_iter):
        x = torch_sparse.spmm(index_pair, vals, N, N, x)
        x = x / (x.norm() + 1e-6)
    y = torch_sparse.spmm(index_pair, vals, N, N, x)
    # Rayleigh quotient
    return ((x * y).sum() / (x * x).sum()).item()

def batched_inv_sqrt_spd(M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute the inverse square root M^{-1/2} for a batch of (approximately) SPD matrices.

    Step-by-step:
      1) Eigendecompose each matrix in the batch: M = V diag(λ) V^T
      2) Clamp eigenvalues from below so λ >= eps to avoid singularities
      3) Recompose M^{-1/2} = V diag(λ^{-1/2}) V^T

    Args:
        M: Tensor of shape [N, d, d], each slice assumed symmetric positive semidefinite
        eps: Minimum eigenvalue used for clamping during inversion

    Returns:
        Tensor of shape [N, d, d] containing the inverse square roots
    """
    if M.dim() != 3 or M.size(-1) != M.size(-2):
        raise ValueError(f"Expected M to be [N,d,d], got {tuple(M.shape)}")

    # Eigendecomposition: eigenvectors form an orthogonal basis, eigenvalues are real.
    evals, evecs = torch.linalg.eigh(M)              # evals: [N,d], evecs: [N,d,d]

    # Clamp eigenvalues to prevent division-by-zero when taking λ^{-1/2}.
    evals = evals.clamp_min(eps)

    # Construct V diag(λ^{-1/2}) V^T (batched) to get the inverse square root.
    inv_sqrt = evecs @ torch.diag_embed(evals.rsqrt()) @ evecs.transpose(-1, -2)
    return inv_sqrt


def batched_procrustes_align(
    Z: torch.Tensor,
    anchor: torch.Tensor,
    proper: bool = True,
) -> torch.Tensor:
    """
    Align each node stalk Z_i ∈ R^{dxC} to a shared anchor frame (a.k.a. Global Frame) via orthogonal Procrustes.

    Steps:
      1) Cross-covariance: C_i = Z_i @ anchor^T  (dxd)
      2) SVD: C_i = U Σ V^T
      3) Orthogonal map: R_i = U V^T (closest rotation/reflection to C_i)
      4) Optional: enforce det(R_i)=+1 by flipping last column of U if det(R_i)<0
      5) Align Z_i: Z_i_aligned = R_i^T @ Z_i so every stalk shares the anchor orientation

    Args:
        Z: Tensor [N, d, C] containing per-node stalk representations
        anchor: Tensor [d, C] defining the global reference frame
        proper: Enforce rotation matrices with det=+1 (no reflections) if True

    Returns:
        Tensor [N, d, C] with all stalks rotated into the common anchor frame
    """
    if Z.dim() != 3:
        raise ValueError(f"Expected Z to be [N,d,C], got {tuple(Z.shape)}")

    N, d, C = Z.shape
    if anchor.shape != (d, C):
        raise ValueError(f"Expected anchor to be [d,C]=({d},{C}), got {tuple(anchor.shape)}")

    # Cross-covariance measures how far each stalk basis drifts from the anchor frame.
    Ccov = Z @ anchor.transpose(0, 1)                # [N,d,d]
    U, _, Vh = torch.linalg.svd(Ccov, full_matrices=False)
    R = U @ Vh                                       # [N,d,d]

    if proper:
        # Enforce det(R)=+1 by flipping the last column of U when det is negative.
        detR = torch.det(R)
        neg = detR < 0
        if neg.any():
            U_fix = U.clone()
            U_fix[neg, :, -1] *= -1.0
            R = U_fix @ Vh

    # Rotate each stalk into the anchor’s coordinate system.
    Z_aligned = R.transpose(-1, -2) @ Z              # [N,d,C]
    return Z_aligned