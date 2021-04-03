import dask.array as da
import numpy as np
import pytest

try:
    import mps  # noqa: F401

    MPS_NOT_FOUND = False
except ImportError:
    MPS_NOT_FOUND = True

from mps_motion_tracking import frame_sequence as fs

array_type = {da: da.core.Array, np: np.ndarray}


@pytest.mark.parametrize("ns", [np, da])
def test_frame_sequence(ns):
    width = 10
    height = 12
    num_time_steps = 14

    x = fs.FrameSequence(ns.random.random((width, height, num_time_steps)))

    x_mean = x.mean()
    assert isinstance(x_mean, array_type[ns])
    assert x.mean().shape == (num_time_steps,)

    x_max = x.max()
    assert isinstance(x_max, array_type[ns])
    assert x_max.shape == (width, height)


@pytest.mark.parametrize("ns", [np, da])
def test_vector_frame_sequence(ns):
    width = 10
    height = 12
    num_time_steps = 14

    arr = ns.random.random((width, height, num_time_steps, 2))

    x = fs.VectorFrameSequence(arr)

    x_mean = x.mean()
    assert isinstance(x_mean, array_type[ns])
    assert x.mean().shape == (num_time_steps, 2)

    x_max = x.max()
    assert isinstance(x_max, array_type[ns])
    assert x_max.shape == (width, height, 2)

    x_norm = x.norm()
    assert isinstance(x_norm, fs.FrameSequence)
    assert x_norm.shape == (width, height, num_time_steps)

    x_x = x.x
    assert isinstance(x_x, fs.FrameSequence)
    assert x_x.shape == (width, height, num_time_steps)
    assert x_x == fs.FrameSequence(arr[:, :, :, 0])

    x_y = x.y
    assert isinstance(x_y, fs.FrameSequence)
    assert x_y.shape == (width, height, num_time_steps)
    assert x_y == fs.FrameSequence(arr[:, :, :, 1])


@pytest.mark.parametrize("ns", [np, da])
def test_tensor_frame_sequence(ns):

    width = 10
    height = 12
    num_time_steps = 14

    arr = ns.ones((width, height, num_time_steps, 2, 2))

    x = fs.TensorFrameSequence(arr)

    x_x = x.x
    assert isinstance(x_x, fs.FrameSequence)
    assert x_x.shape == (width, height, num_time_steps)
    assert x_x == fs.FrameSequence(arr[:, :, :, 0, 0])

    x_y = x.y
    assert isinstance(x_y, fs.FrameSequence)
    assert x_y.shape == (width, height, num_time_steps)
    assert x_y == fs.FrameSequence(arr[:, :, :, 1, 1])

    x_xy = x.xy
    assert isinstance(x_xy, fs.FrameSequence)
    assert x_xy.shape == (width, height, num_time_steps)
    assert x_xy == fs.FrameSequence(arr[:, :, :, 1, 0])

    x_norm = x.norm()
    assert isinstance(x_norm, fs.FrameSequence)
    assert x_norm.shape == (width, height, num_time_steps)

    x_mean = x.mean()
    assert isinstance(x_mean, array_type[ns])
    assert x.mean().shape == (num_time_steps, 2, 2)

    x_max = x.max()
    assert isinstance(x_max, array_type[ns])
    assert x_max.shape == (width, height, 2, 2)

    try:
        # If we have a dask array
        a = arr.compute()
    except AttributeError:
        a = arr

    q = np.linalg.eigvalsh(a)
    p = x.compute_eigenvalues()
    assert p.shape == q.shape
    assert np.isclose(p.array, q).all()


@pytest.mark.skipif(MPS_NOT_FOUND, reason="MPS not found")
@pytest.mark.parametrize("ns", [np, da])
def test_local_averages(ns):

    width = 10
    height = 15
    num_time_steps = 14

    arr = ns.ones((width, height, num_time_steps))

    x = fs.FrameSequence(arr)

    N = 3

    la = x.local_averages(N)
    assert la.shape == (width // (height // N), N, num_time_steps)


if __name__ == "__main__":
    test_local_averages(np)
