from pathlib import Path

import numpy as np
import pytest

from mps_motion_tracking.utils import MPSData

here = Path(__file__).absolute().parent


@pytest.fixture
def test_data():
    return MPSData(
        **np.load(here.joinpath("../datasets/mps_data.npy"), allow_pickle=True).item()
    )
