import pytest
import torch
from torch_geometric.data import Data, HeteroData
from polynsd.datasets.utils import SubsetTrainSplit

# Assuming SubsetTrainSplit is imported or defined in the same file


@pytest.fixture
def homo_data():
    """Fixture to provide a standard homogeneous graph."""
    return Data(
        x=torch.randn(100, 16),
        train_mask=torch.ones(100, dtype=torch.bool),
        test_mask=torch.ones(100, dtype=torch.bool),
    )


@pytest.fixture
def hetero_data():
    """Fixture to provide a standard heterogeneous graph."""
    data = HeteroData()
    data["user"].x = torch.randn(50, 8)
    data["user"].train_mask = torch.ones(50, dtype=torch.bool)
    data["item"].x = torch.randn(50, 8)
    data["item"].train_mask = torch.ones(50, dtype=torch.bool)
    data["item"].test_mask = torch.ones(50, dtype=torch.bool)
    return data


@pytest.mark.parametrize("use_index", [True, False])
def test_split_ratio_and_preservation(homo_data, use_index):
    """Verifies that the split ratio is accurate and test_mask is untouched."""
    val_ratio = 0.2
    transform = SubsetTrainSplit(val_ratio=val_ratio, use_index=use_index, seed=42)
    out = transform(homo_data)

    if use_index:
        assert len(out.train_idx) == 80
        assert len(out.val_idx) == 20
    else:
        assert out.train_mask.sum().item() == 80
        assert out.val_mask.sum().item() == 20
        # Ensure no overlap
        assert not torch.any(out.train_mask & out.val_mask)

    # Critical: Ensure test_mask was not modified
    assert out.test_mask.sum().item() == 100


def test_hetero_node_types(hetero_data):
    """Ensures transform applies to all node types containing a train_mask."""
    transform = SubsetTrainSplit(val_ratio=0.1, use_index=False, seed=42)
    out = transform(hetero_data)

    # Check 'user' type
    assert out["user"].train_mask.sum().item() == 45
    assert out["user"].val_mask.sum().item() == 5

    # Check 'item' type
    assert out["item"].train_mask.sum().item() == 45
    assert out["item"].test_mask.sum().item() == 50


def test_reproducibility(homo_data):
    """Ensures identical seeds produce identical splits."""
    t1 = SubsetTrainSplit(val_ratio=0.2, seed=123)
    t2 = SubsetTrainSplit(val_ratio=0.2, seed=123)

    out1 = t1(homo_data.clone())
    out2 = t2(homo_data.clone())

    assert torch.equal(out1.train_mask, out2.train_mask)


def test_no_train_mask_error():
    """Tests that the transform raises an error if no train_mask is present."""
    data = Data(x=torch.randn(10, 10))
    transform = SubsetTrainSplit()
    with pytest.raises(
        ValueError, match="Data object does not have any 'train_mask' to split."
    ):
        transform(data)
