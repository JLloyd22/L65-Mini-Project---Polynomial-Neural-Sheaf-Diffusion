import torch
import pytest


@pytest.fixture(scope="session")
def device():
    """
    Detects the best available hardware once per test session.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    # elif torch.backends.mps.is_available(): # Support for Apple Silicon GPUs
    #     return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def dtype():
    """
    Provides a consistent precision for tests.
    Can be easily swapped to torch.float64 for debugging numerical stability.
    """
    return torch.float32


@pytest.fixture
def default_setup(device, dtype):
    """
    A helper fixture that returns a dictionary of common settings.
    """
    return {"device": device, "dtype": dtype}
