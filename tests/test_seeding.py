import random

import numpy as np
import pytest
import torch

from tiny_dreamer_highway.utils import set_global_seeds


def test_set_global_seeds_repeats_python_numpy_and_torch_streams() -> None:
    set_global_seeds(7)
    sample_a = (
        random.random(),
        np.random.rand(3),
        torch.rand(3),
    )

    set_global_seeds(7)
    sample_b = (
        random.random(),
        np.random.rand(3),
        torch.rand(3),
    )

    assert sample_a[0] == sample_b[0]
    assert np.allclose(sample_a[1], sample_b[1])
    assert torch.allclose(sample_a[2], sample_b[2])


def test_set_global_seeds_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        set_global_seeds(-1)