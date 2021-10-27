import dask.array as da
import numpy as np
import pytest

from mps_motion_tracking import utils


@pytest.mark.parametrize(
    "filter_type, size, sigma",
    [
        (utils.Filters.median, 3, None),
        (utils.Filters.gaussian, None, 1),
    ],
)
def test_filter_vectors_numpy(filter_type, size, sigma):

    shape = (10, 9, 2)
    np.random.seed(1)
    vectors = 10 * np.ones(shape) + np.random.random(shape)

    filtered_vectors = utils.filter_vectors(
        vectors,
        filter_type=filter_type,
        size=size,
        sigma=sigma,
    )
    assert filtered_vectors.shape == shape

    assert 0 < np.abs(filtered_vectors - vectors).max() < 1


@pytest.mark.parametrize(
    "filter_type, size, sigma",
    [
        (utils.Filters.median, None, None),
        (utils.Filters.gaussian, None, None),
        (utils.Filters.median, 0, None),
        (utils.Filters.gaussian, None, 0),
        (utils.Filters.median, None, 3),
        (utils.Filters.gaussian, 1, None),
        ("somefilter", None, None),
    ],
)
def test_filter_with_0_do_nothing(filter_type, size, sigma):
    shape = (10, 9, 2)
    np.random.seed(1)
    vectors = 10 * np.ones(shape) + np.random.random(shape)

    filtered_vectors = utils.filter_vectors(
        vectors,
        filter_type=filter_type,
        size=size,
        sigma=sigma,
    )
    assert filtered_vectors.shape == shape

    assert np.isclose(filtered_vectors, vectors).all()


@pytest.mark.parametrize(
    "filter_type, size, sigma",
    [
        (utils.Filters.median, 3, None),
        (utils.Filters.gaussian, None, 1),
    ],
)
def test_filter_vectors_dask(filter_type, size, sigma):

    shape = (10, 9, 2)
    np.random.seed(1)
    vectors = 10 * da.ones(shape) + da.random.random(shape)

    filtered_vectors = utils.filter_vectors(
        vectors,
        filter_type=filter_type,
        size=size,
        sigma=sigma,
    )
    assert filtered_vectors.shape == shape
    assert 0 < da.absolute(filtered_vectors - vectors).max().compute() < 1


@pytest.mark.parametrize(
    "filter_type, size, sigma",
    [
        (utils.Filters.median, 3, None),
        (utils.Filters.gaussian, None, 1),
    ],
)
def test_filter_vectors_par_numpy(filter_type, size, sigma):

    shape = (10, 10, 5, 2)
    np.random.seed(1)
    vectors = 10 * np.ones(shape) + np.random.random(shape)

    filtered_vectors = utils.filter_vectors_par(
        vectors,
        filter_type=filter_type,
        size=size,
        sigma=sigma,
    )
    assert filtered_vectors.shape == shape
    assert 0 < np.abs(filtered_vectors - vectors).max() < 1
