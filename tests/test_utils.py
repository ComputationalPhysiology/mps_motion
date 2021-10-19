import time

import dask.array as da
import numpy as np

from mps_motion_tracking import utils


def test_filter_vectors_numpy():

    shape = (3, 4, 2)
    np.random.seed(1)
    vectors = 10 * np.ones(shape) + np.random.random(shape)

    filtered_vectors = utils.filter_vectors(vectors, 5)
    assert filtered_vectors.shape == shape
    assert 0 < np.abs(filtered_vectors - vectors).max() < 1


def test_filter_vectors_dask():

    shape = (3, 4, 2)
    np.random.seed(1)
    vectors = 10 * da.ones(shape) + da.random.random(shape)

    filtered_vectors = utils.filter_vectors(vectors, 5)
    assert filtered_vectors.shape == shape
    assert 0 < da.absolute(filtered_vectors - vectors).max().compute() < 1


def test_filter_vectors_par_numpy():

    shape = (500, 500, 100, 2)
    np.random.seed(1)
    vectors = 10 * np.ones(shape) + np.random.random(shape)

    t0 = time.perf_counter()
    filtered_vectors = utils.filter_vectors_par(vectors, 5)
    print(f"Elpased time {time.perf_counter() - t0}")
    assert filtered_vectors.shape == shape
    assert 0 < np.abs(filtered_vectors - vectors).max() < 1
