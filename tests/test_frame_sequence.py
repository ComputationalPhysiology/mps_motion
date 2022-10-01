from itertools import product
from pathlib import Path

import dask.array as da
import numpy as np
import pytest

try:
    import mps  # noqa: F401

    MPS_NOT_FOUND = False
except ImportError:
    MPS_NOT_FOUND = True

from mps_motion import frame_sequence as fs
from mps_motion import filters

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

    arr = ns.random.random((width, height, num_time_steps, 2, 2))

    x = fs.TensorFrameSequence(arr)

    x_x = x.x
    assert isinstance(x_x, fs.FrameSequence)
    assert x_x.shape == (width, height, num_time_steps)
    assert x_x == fs.FrameSequence(arr[:, :, :, 1, 1])

    x_y = x.y
    assert isinstance(x_y, fs.FrameSequence)
    assert x_y.shape == (width, height, num_time_steps)
    assert x_y == fs.FrameSequence(arr[:, :, :, 0, 0])

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


@pytest.mark.parametrize("ns, suffix", product([np, da], [".h5", ".npy"]))
def test_save_load(ns, suffix):

    width = 10
    height = 15
    num_time_steps = 14

    arr = ns.ones((width, height, num_time_steps))

    x = fs.FrameSequence(arr)
    path = Path("test").with_suffix(suffix)
    x.save(path)

    new_x = fs.FrameSequence.from_file(path)

    assert x == new_x
    path.unlink()


@pytest.mark.parametrize("ns", [np, da])
def test_norm(ns):

    width = 10
    height = 15
    num_time_steps = 14

    arr = ns.ones((width, height, num_time_steps, 2))

    x = fs.VectorFrameSequence(arr)
    x_norm = x.norm()
    assert x_norm.shape == (width, height, num_time_steps)
    assert isinstance(x_norm, fs.FrameSequence)
    arr = x_norm.compute()
    assert (arr == np.sqrt(2)).all()
    # breakpoint()


@pytest.mark.parametrize(
    "ns, limits",
    product([np, da], [(0.4, 0.6), (None, 0.6), (0.4, None), (0.5, 0.5), (None, None)]),
)
def test_threshold(ns, limits):
    vmin, vmax = limits

    width = 10
    height = 15
    num_time_steps = 14

    np.random.seed(1)
    values = ns.random.random((width, height, num_time_steps))

    # This value should remain the same
    special_value = 0.5
    # Norm of this is about 0.56
    values[0, 0, 0] = special_value

    arr = fs.FrameSequence(values)
    th_arr = arr.threshold(vmin, vmax)
    if vmin is None:
        vmin = values.min()
    if vmax is None:
        vmax = values.max()

    assert ns.isclose(th_arr.max().max(), vmax)
    assert ns.isclose(th_arr.min().min(), vmin)
    assert ns.isclose(th_arr.array[0, 0, 0], special_value)


@pytest.mark.parametrize("ns", [np, da])
def test_invalid_threshold_raises(ns):

    width = 10
    height = 15
    num_time_steps = 14

    values = ns.random.random((width, height, num_time_steps))
    arr = fs.FrameSequence(values)

    with pytest.raises(filters.InvalidThresholdError):
        arr.threshold(0.7, 0.2)


@pytest.mark.parametrize(
    "ns, limits",
    product([np, da], [(0.4, 0.6), (None, 0.6), (0.4, None), (0.5, 0.5), (None, None)]),
)
def test_threshold_norm(ns, limits):
    vmin, vmax = limits

    width = 10
    height = 15
    num_time_steps = 14

    np.random.seed(1)
    values = ns.random.random((width, height, num_time_steps, 2))

    # This value should remain the same
    special_value = np.sqrt(0.125)
    # sqrt(0.125 + 0.125) = sqrt(0.25) = 0.5
    values[0, 0, 0, 0] = special_value
    values[0, 0, 0, 1] = special_value

    arr = fs.VectorFrameSequence(values)
    th_arr = arr.threshold_norm(vmin, vmax)
    if vmin is None:
        vmin = arr.norm().min().min()
    if vmax is None:
        vmax = arr.norm().max().max()

    assert ns.isclose(th_arr.norm().max().max(), vmax)
    assert ns.isclose(th_arr.norm().min().min(), vmin)
    assert ns.isclose(th_arr.array[0, 0, 0, 0], special_value)
    assert ns.isclose(th_arr.array[0, 0, 0, 1], special_value)


@pytest.mark.parametrize(
    "filter_type, size, sigma",
    [
        (filters.Filters.median, 3, None),
        (filters.Filters.gaussian, None, 1),
    ],
)
def test_filter_VectorFrameSequence(filter_type, size, sigma):

    shape = (10, 9, 8, 2)
    np.random.seed(1)
    vectors = 10 * np.ones(shape) + np.random.random(shape)

    u = fs.VectorFrameSequence(vectors)

    u_filt = u.filter(filter_type=filter_type, size=size, sigma=sigma)
    filtered_vectors = u_filt.array

    assert filtered_vectors.shape == shape

    assert 0 < np.abs(filtered_vectors - vectors).max() < 1


@pytest.mark.parametrize(
    "filter_type, size, sigma",
    [
        (filters.Filters.median, 3, None),
        (filters.Filters.gaussian, None, 1),
    ],
)
def test_filter_FrameSequence(filter_type, size, sigma):

    shape = (10, 9, 8)
    np.random.seed(1)
    vectors = 10 * np.ones(shape) + np.random.random(shape)

    u = fs.FrameSequence(vectors)

    u_filt = u.filter(filter_type=filter_type, size=size, sigma=sigma)
    filtered_vectors = u_filt.array

    assert filtered_vectors.shape == shape

    assert 0 < np.abs(filtered_vectors - vectors).max() < 1


def test_cartToPolar():

    # a = 0.01
    # b = 0.01
    # def func(x):
    #     return (x[0] * (1 - a), x[1] * (1 - b))

    # frame1 = np.random.randint(size=(12, 12))
    # from scipy.ndimage import geometric_transform
    # frame2 = geometric_transform(frame1.T, func).T
    # np.fromfunction
    pass


if __name__ == "__main__":
    test_threshold_norm(da, (0.4, 0.6))
