import torch
import pytest


@pytest.fixture(autouse=True)
def deterministic_test():
    torch.set_num_threads(2)
    torch.manual_seed(123)
