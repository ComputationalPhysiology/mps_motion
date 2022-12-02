import typing
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from mps_motion import frame_sequence as fs
from mps_motion import Mechanics
from mps_motion import MPSData
from mps_motion import utils
from scipy.ndimage import geometric_transform

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


@pytest.fixture
def mech_obj():

    width = 10
    height = 15
    num_time_points = 4

    u = np.zeros((height, width, num_time_points, 2))
    # First time point is zero
    # Second time point has a linar displacement in x
    u[:, :, 1, 0] = np.fromfunction(
        lambda y, x: x / width,
        shape=(height, width),
        dtype=float,
    )
    # Third points have linear displacement in y
    u[:, :, 2, 1] = np.fromfunction(
        lambda y, x: y / height,
        shape=(height, width),
        dtype=float,
    )
    # Forth is linear in both
    u[:, :, 2, 0] = u[:, :, 1, 0]
    u[:, :, 3, 1] = u[:, :, 2, 1]

    return Mechanics(fs.VectorFrameSequence(u), t=1000 * np.arange(4))


@pytest.fixture
def mech_trace_obj():
    traces = np.load(here.joinpath("example_traces.npy"), allow_pickle=True).item()

    mechanics_mock = mock.Mock(spec=Mechanics)
    mechanics_mock.u.norm().mean.return_value = traces["u_norm"]
    mechanics_mock.velocity().norm().mean.return_value = traces["v_norm"]
    mechanics_mock.t = traces["time"]
    return mechanics_mock


class SyntheticTrace(typing.NamedTuple):
    t: np.ndarray
    u: np.ndarray
    v: np.ndarray


@pytest.fixture
def synthetic_trace():
    t = np.arange(0, 1, 0.01)
    u = utils.ca_transient(t)
    v = np.abs(np.diff(u))
    return SyntheticTrace(t=t, u=u, v=v)
