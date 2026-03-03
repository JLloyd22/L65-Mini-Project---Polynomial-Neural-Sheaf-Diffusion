"""
Compatibility layer for torch_sparse using native PyTorch sparse operations.
This module provides a drop-in replacement for torch_sparse.spmm when torch_sparse is not available.
"""

import torch


def spmm(index, value, m, n, tensor):
    """
    Sparse matrix multiplication: Computes A @ tensor, where A is sparse in COO format.
    
    Args:
        index: Tensor of shape (2, nnz) containing row and column indices
        value: Tensor of shape (nnz,) containing the values
        m: Number of rows in the sparse matrix
        n: Number of columns in the sparse matrix  
        tensor: Dense tensor for right-multiplication
        
    Returns:
        Result of sparse matrix multiplication
    """
    # Handle sparse matrix format
    # index is (2, nnz) for COO format, value is (nnz,)
    
    # Create sparse matrix and convert tensor to appropriate device
    if not isinstance(index, torch.Tensor):
        index = torch.tensor(index, device=tensor.device)
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=tensor.dtype, device=tensor.device)
    
    # Ensure we're on the same device
    index = index.to(device=tensor.device, dtype=torch.long)
    value = value.to(dtype=tensor.dtype, device=tensor.device)
    
    # Create sparse matrix in COO format
    sparse_matrix = torch.sparse_coo_tensor(index, value, (m, n), device=tensor.device, dtype=tensor.dtype)
    
    # Perform sparse-dense matrix multiplication
    # torch.sparse.mm expects (sparse, dense) -> dense
    if tensor.dim() == 1:
        # For vector input, reshape to (n, 1), multiply, then reshape back
        tensor_2d = tensor.unsqueeze(1)
        result = torch.sparse.mm(sparse_matrix, tensor_2d)
        return result.squeeze(1)
    else:
        # For matrix input, multiply directly
        return torch.sparse.mm(sparse_matrix, tensor)


__all__ = ['spmm']
