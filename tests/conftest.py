from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import geometric_transform

from mps_motion_tracking.utils import MPSData

here = Path(__file__).absolute().parent

_TESTFILE_NAME = here.joinpath("../datasets/mps_data.npy").as_posix()


@pytest.fixture
def TEST_FILENAME():
    return _TESTFILE_NAME


def get_func(t, a=0.001, b=0):
    def func(x):
        return (x[0] * (1 - a * np.sin(t)), x[1] * (1 - b * np.sin(t)))

    return func


@pytest.fixture
def test_data():

    N = 10
    a = 0.0
    b = 0.02
    frame = np.load(here.joinpath("../datasets/first_frame.npy"))
    frames = [frame]
    times = np.linspace(0, np.pi, N)
    for t in times[1:]:
        frames.append(geometric_transform(frame, get_func(t, a, b)))

    frames = np.array(frames).T
    info = {
        "num_frames": N,
        "dt": np.diff(times).mean(),
        "time_unit": "ms",
        "um_per_pixel": 3.25,
        "size_x": frames.shape[0],
        "size_y": frames.shape[1],
    }

    return MPSData(frames=frames, time_stamps=times, info=info)


# @pytest.fixture
# def test_data():
# return MPSData(**np.load(_TESTFILE_NAME, allow_pickle=True).item())
