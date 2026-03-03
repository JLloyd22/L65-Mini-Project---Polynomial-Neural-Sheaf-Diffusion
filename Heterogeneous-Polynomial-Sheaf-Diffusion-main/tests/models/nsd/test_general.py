import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph
from polynsd.models.nsd.general import GeneralNSDConv


@pytest.fixture
def sample_graph():
    """Creates a small random graph for testing."""
    num_nodes = 10
    in_channels = 32
    stalk_dim = 4

    # Generate random features
    x_feat = torch.randn(num_nodes, in_channels)
    x_stalk = torch.randn(num_nodes, stalk_dim, 1)

    # Generate random edges (Erdos-Renyi)
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.3)

    return x_feat, x_stalk, edge_index


def test_output_shape(sample_graph):
    """Verify that the output stalk has the same shape as the input stalk."""
    x_feat, x_stalk, edge_index = sample_graph
    in_channels = x_feat.size(1)
    stalk_dim = x_stalk.size(1)

    conv = GeneralNSDConv(d=stalk_dim, in_channels=in_channels, hidden_dim=16)

    out = conv(x_feat, x_stalk, edge_index)

    assert out.shape == x_stalk.shape
    assert not torch.isnan(out).any(), "Output contains NaNs"


def test_permutation_invariance(sample_graph):
    """GNNs must yield the same result regardless of node ordering."""
    x_feat, x_stalk, edge_index = sample_graph
    stalk_dim = x_stalk.size(1)
    in_channels = x_feat.size(1)

    conv = GeneralNSDConv(d=stalk_dim, in_channels=in_channels, hidden_dim=16)
    conv.eval()

    # Original pass
    out_orig = conv(x_feat, x_stalk, edge_index)

    # Create permutation
    perm = torch.randperm(x_feat.size(0))
    rev_perm = torch.argsort(perm)

    # Permute inputs
    x_feat_p = x_feat[perm]
    x_stalk_p = x_stalk[perm]

    # Adjust edge_index for permutation
    # Map old indices to new indices
    mapping = {old.item(): new for new, old in enumerate(perm)}
    edge_index_p = edge_index.clone()
    for i in range(edge_index.size(1)):
        edge_index_p[0, i] = mapping[edge_index[0, i].item()]
        edge_index_p[1, i] = mapping[edge_index[1, i].item()]

    # Permuted pass
    out_perm = conv(x_feat_p, x_stalk_p, edge_index_p)

    # Revert permutation on output
    out_unpermuted = out_perm[rev_perm]

    torch.testing.assert_close(out_orig, out_unpermuted, atol=1e-5, rtol=1e-5)


def test_gradient_flow(sample_graph):
    """Ensure the MLP parameters receive gradients."""
    x_feat, x_stalk, edge_index = sample_graph
    stalk_dim = x_stalk.size(1)
    in_channels = x_feat.size(1)

    conv = GeneralNSDConv(d=stalk_dim, in_channels=in_channels, hidden_dim=16)

    out = conv(x_feat, x_stalk, edge_index)
    loss = out.pow(2).sum()
    loss.backward()

    # Check if MLP weights have gradients
    for name, param in conv.map_generator.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.abs(param.grad).sum() > 0, f"Zero gradient for {name}"


def test_disconnected_node():
    """Test behavior when a node has no neighbors (isolated node)."""
    num_nodes = 3
    d = 2
    in_c = 4

    x_feat = torch.randn(num_nodes, in_c)
    x_stalk = torch.randn(num_nodes, d, 1)
    # Only one edge between 0 and 1; node 2 is isolated
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    conv = GeneralNSDConv(d=d, in_channels=in_c, hidden_dim=8)
    out = conv(x_feat, x_stalk, edge_index)

    # For an isolated node (node 2), the update should result in the original stalk
    # because diag_part and adj_out should both be zero.
    assert torch.allclose(out[2], x_stalk[2]), "Isolated node stalk was modified"
