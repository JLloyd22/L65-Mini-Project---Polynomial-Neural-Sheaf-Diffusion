#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT
import torch
from torch.testing import assert_close
from polynsd.models.sheaf_gnn.sheaf_models import TypeEnsembleSheafLearner


def test_type_ensemble_robustness():
    # Use double precision to eliminate floating point accumulation noise
    dtype = torch.float64
    d_in, d_out, num_types = 5, 5, 4

    x = torch.rand(10, d_in, dtype=dtype)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)

    # Intentionally skip some types (e.g., only types 0 and 3)
    edge_type = torch.tensor([0, 3, 0, 3, 0], dtype=torch.long)
    node_type = torch.randint(0, num_types, (10,))

    module = TypeEnsembleSheafLearner(d_in, (d_out,), "id", num_types, num_types).to(
        dtype
    )
    module.eval()

    with torch.no_grad():
        # Optimized forward pass
        out_optimized = module(x, edge_index, edge_type, node_type)

        # Manual Ground Truth
        src, dst = edge_index
        x_cat = torch.cat([x[src], x[dst]], dim=1)

        manual_results = []
        for i in range(edge_index.size(1)):
            etype = edge_type[i].item()
            manual_results.append(module.linear1[etype](x_cat[i]))

        out_manual = torch.stack(manual_results)

    # Verification
    # Using 1e-10 because we are in float64
    assert_close(out_optimized, out_manual, atol=1e-10, rtol=1e-10)


def test_type_ensemble_shapes():
    """Verify 2D out_shape (d, d) for matrix-based sheaves."""
    d_in, d_out = 4, 2
    module = TypeEnsembleSheafLearner(d_in, (d_out, d_out), "id", 2, 2)
    x = torch.randn(5, d_in)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    edge_type = torch.tensor([0, 1])

    out = module(x, edge_index, edge_type)
    assert out.shape == (2, d_out, d_out)
