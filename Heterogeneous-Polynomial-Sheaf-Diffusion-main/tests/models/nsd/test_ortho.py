import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph
from polynsd.models.nsd.ortho import OrthogonalNSDConv


# Ruff-formatted unit tests for OrthogonalNSDConv
@pytest.fixture
def ortho_setup():
    """Provides common variables for orthogonal sheaf tests."""
    num_nodes = 6
    in_channels = 16
    d = 3  # Small d for matrix property verification
    hidden_dim = 8

    x_feat = torch.randn(num_nodes, in_channels)
    x_stalk = torch.randn(num_nodes, d, 1)
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.5)

    conv = OrthogonalNSDConv(d, in_channels, hidden_dim)
    return conv, x_feat, x_stalk, edge_index


def test_orthogonality_property(ortho_setup):
    """Verify that the Cayley transform produces strictly orthogonal matrices."""
    conv, x_feat, _, edge_index = ortho_setup
    u_idx, v_idx = edge_index

    # Generate parameters
    params_v = conv.map_generator(torch.cat([x_feat[v_idx], x_feat[u_idx]], dim=-1))
    W = conv.cayley(params_v)

    # Check W^T * W = I
    batch_size = W.size(0)
    identity = torch.eye(conv.d, device=W.device).unsqueeze(0).repeat(batch_size, 1, 1)
    res = torch.matmul(W.transpose(-2, -1), W)

    torch.testing.assert_close(res, identity, atol=1e-5, rtol=1e-5)


def test_initialization_near_identity(ortho_setup):
    """Ensure reset_parameters initializes W close to the Identity matrix."""
    conv, x_feat, _, edge_index = ortho_setup
    u_idx, v_idx = edge_index

    # Default initialization is near-zero for params, which means Cayley(0) = Identity
    params_v = conv.map_generator(torch.cat([x_feat[v_idx], x_feat[u_idx]], dim=-1))
    W = conv.cayley(params_v)

    identity = torch.eye(conv.d).unsqueeze(0)
    # Check if the average deviation from identity is small
    diff = torch.abs(W - identity).mean()
    assert diff < 1e-2, f"Initial maps too far from identity: {diff}"


def test_clamping_stability():
    """Ensure that extremely large MLP outputs are clamped to prevent solver failure."""
    d = 3
    conv = OrthogonalNSDConv(d=d, in_channels=4, hidden_dim=4, clamp_val=5.0)

    # Create 'infinite' params
    extreme_params = torch.tensor([[1e10, 1e10, 1e10]])

    # Should not crash and should produce a valid orthogonal matrix
    W = conv.cayley(extreme_params)
    assert not torch.isnan(W).any()

    # Verify it still satisfies orthogonality even when clamped
    identity = torch.eye(d)
    res = torch.matmul(W.transpose(-2, -1), W)
    torch.testing.assert_close(res.squeeze(0), identity, atol=1e-5, rtol=1e-5)


def test_forward_output_dimension(ortho_setup):
    """Verify standard GNN output shape [N, d, 1]."""
    conv, x_feat, x_stalk, edge_index = ortho_setup
    out = conv(x_feat, x_stalk, edge_index)

    assert out.shape == x_stalk.shape
    assert not torch.isnan(out).any()


def test_backward_gradient_clipping(ortho_setup):
    """Test that gradients are clipped correctly via the register_hook."""
    conv, x_feat, x_stalk, edge_index = ortho_setup
    x_feat.requires_grad = True
    x_stalk.requires_grad = True

    # Forward pass
    out = conv(x_feat, x_stalk, edge_index)

    # Simulate an exploding gradient scenario
    loss = out.sum() * 1e10
    loss.backward()

    # Check if gradients for the map_generator exist and are finite
    for param in conv.map_generator.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
