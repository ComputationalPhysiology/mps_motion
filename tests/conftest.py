from pathlib import Path

import numpy as np
import pytest

from mps_motion_tracking.utils import MPSData

here = Path(__file__).absolute().parent

TESTFILE_NAME = here.joinpath("../datasets/mps_data.npy").as_posix()


@pytest.fixture
def test_data():
    return MPSData(**np.load(TESTFILE_NAME, allow_pickle=True).item())
