import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph
from polynsd.models.nsd.diagonal import DiagonalNSDConv


@pytest.fixture
def diag_setup():
    """Common variables for diagonal sheaf tests."""
    num_nodes = 8
    in_channels = 16
    d = 4
    hidden_dim = 8

    x_feat = torch.randn(num_nodes, in_channels)
    x_stalk = torch.randn(num_nodes, d, 1)
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)

    conv = DiagonalNSDConv(d, in_channels, hidden_dim)
    return conv, x_feat, x_stalk, edge_index


def test_diagonal_output_shape(diag_setup):
    """Verify the output maintains [N, d, 1] shape."""
    conv, x_feat, x_stalk, edge_index = diag_setup
    out = conv(x_feat, x_stalk, edge_index)

    assert out.shape == x_stalk.shape
    assert not torch.isnan(out).any()


def test_diagonal_elementwise_logic(diag_setup):
    """
    Verify that channels are treated independently.
    Changing channel i in x_stalk should not affect channel j in the output.
    """
    conv, x_feat, x_stalk, edge_index = diag_setup

    # Pass 1
    out1 = conv(x_feat, x_stalk, edge_index)

    # Pass 2: Perturb only the first channel of the input stalk
    x_stalk_perturbed = x_stalk.clone()
    x_stalk_perturbed[:, 0, :] += 1.0
    out2 = conv(x_feat, x_stalk_perturbed, edge_index)

    # Channels 1 through d should be identical to the original pass
    # Only channel 0 should differ.
    torch.testing.assert_close(out1[:, 1:, :], out2[:, 1:, :])
    assert not torch.allclose(out1[:, 0, :], out2[:, 0, :])


def test_diagonal_gradient_flow(diag_setup):
    """Check that gradients propagate through the element-wise squares and products."""
    conv, x_feat, x_stalk, edge_index = diag_setup

    out = conv(x_feat, x_stalk, edge_index)
    loss = out.pow(2).sum()
    loss.backward()

    for name, param in conv.map_generator.named_parameters():
        assert param.grad is not None
        assert torch.abs(param.grad).sum() > 0


def test_diagonal_zero_weights(diag_setup):
    """
    Test edge case: if maps are zero, the update should return original stalk.
    h = h + alpha * (0 - 0) = h
    """
    conv, x_feat, x_stalk, edge_index = diag_setup

    # Force the MLP to output zeros
    with torch.no_grad():
        for layer in conv.map_generator:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()

    out = conv(x_feat, x_stalk, edge_index)
    torch.testing.assert_close(out, x_stalk)


def test_diagonal_normalization_scaling(diag_setup):
    """Verify that normalization coefficients are applied correctly."""
    conv, x_feat, x_stalk, edge_index = diag_setup

    # If alpha is 0, the output must be exactly the input stalk
    conv.alpha.data.zero_()
    out = conv(x_feat, x_stalk, edge_index)
    torch.testing.assert_close(out, x_stalk)
