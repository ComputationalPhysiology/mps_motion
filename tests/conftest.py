from pathlib import Path

import numpy as np
import pytest

from mps_motion_tracking.utils import MPSData

here = Path(__file__).absolute().parent

_TESTFILE_NAME = here.joinpath("../datasets/mps_data.npy").as_posix()


@pytest.fixture
def TEST_FILENAME():
    return _TESTFILE_NAME


@pytest.fixture
def test_data():
    return MPSData(**np.load(_TESTFILE_NAME, allow_pickle=True).item())
