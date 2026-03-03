"""
Compatibility layer for torch_scatter using native PyTorch scatter operations.
This module provides drop-in replacements for torch_scatter functions when torch_scatter is not available.
"""

import torch


def scatter_add(src, index, dim_size=None, dim=0):
    """
    Scatters values from src tensor along a dimension, accumulating values.
    
    Args:
        src: Source tensor
        index: Indices tensor specifying where to scatter
        dim_size: Size of the dimension to scatter to (optional, auto-determined if None)
        dim: Dimension along which to scatter (default: 0)
        
    Returns:
        Scattered tensor with accumulated values
    """
    if dim_size is None:
        dim_size = index.max().item() + 1 if index.numel() > 0 else 0
    
    # Use torch_scatter_add alternative
    # Create output tensor
    shape = list(src.shape)
    shape[dim] = dim_size
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    
    # Perform scatter add using index_add_
    if dim == 0:
        out.index_add_(0, index, src)
    else:
        # For other dimensions, we need to use a different approach
        # Move the scatter dimension to the front, scatter, then move back
        perm = list(range(src.dim()))
        perm[0], perm[dim] = perm[dim], perm[0]
        src_moved = src.permute(*perm)
        out_moved = out.permute(*perm)
        out_moved.index_add_(0, index, src_moved)
        out = out_moved.permute(*perm)
    
    return out


# Aliases for common PyTorch scatter operations that might be used
scatter = scatter_add


__all__ = ['scatter_add', 'scatter']
