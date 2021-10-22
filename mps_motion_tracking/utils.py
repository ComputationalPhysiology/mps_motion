import logging
import os
from enum import Enum
from typing import Optional
from typing import Union

import dask
import dask.array as da
import dask_image.ndfilters
import numpy as np
from dask.diagnostics import ProgressBar

logger = logging.getLogger(__name__)

try:
    from numba import jit, prange
except ImportError:
    msg = (
        "numba not found - Numba is just to speed up the motion tracking algorithm\n"
        "To install numba use: pip install numba"
    )
    logger.debug(msg)
    prange = range

    # Create a dummy decorator
    def jit(**params):
        def decorator(f):
            def wrap(*args, **kwargs):
                return f(*args, **kwargs)

            return wrap

        return decorator


class ShapeError(RuntimeError):
    pass


PathLike = Union[str, os.PathLike]
Array = Union[da.core.Array, np.ndarray]


class Filters(str, Enum):
    median = "median"
    gaussian = "gaussian"


def filter_vectors(
    vectors: Array,
    filter_type: Filters,
    size: Optional[int] = None,
    sigma: Optional[float] = None,
):

    if not valid_filter(filter_type=filter_type, size=size, sigma=sigma):
        return vectors

    is_numpy = False
    if isinstance(vectors, np.ndarray):
        is_numpy = True
        vectors = da.from_array(vectors)

    vec0 = apply_filter(
        vectors[:, :, 0],
        filter_type=filter_type,
        size=size,
        sigma=sigma,
    )
    vec1 = apply_filter(
        vectors[:, :, 1],
        filter_type=filter_type,
        size=size,
        sigma=sigma,
    )
    vectors = da.stack([vec0, vec1], axis=-1)
    if is_numpy:
        vectors = vectors.compute()
    return vectors


def filter_vectors_par(
    vectors,
    filter_type: Filters,
    size: Optional[int] = None,
    sigma: Optional[float] = None,
):

    if not valid_filter(filter_type=filter_type, size=size, sigma=sigma):
        return vectors
    logger.info("Filter vectors")
    assert len(vectors.shape) == 4
    assert vectors.shape[3] == 2
    num_frames = vectors.shape[2]

    is_numpy = False
    if isinstance(vectors, np.ndarray):
        is_numpy = True
        vectors = da.from_array(vectors)

    all_vectors = []
    for i in range(num_frames):
        all_vectors.append(
            dask.delayed(filter_vectors)(
                vectors[:, :, i, :],
                filter_type=filter_type,
                size=size,
                sigma=sigma,
            ),
        )

    with ProgressBar():
        vectors = da.stack(*da.compute(all_vectors), axis=2)
    if is_numpy:
        vectors = vectors.compute()

    logger.info("Done filtering")
    return vectors


def valid_filter(
    filter_type: Filters,
    size: Optional[int] = None,
    sigma: Optional[float] = None,
) -> bool:
    if filter_type == Filters.median and size is None:
        logger.warning("Please provide a size of the median filter kernel")
        return False
    elif filter_type == Filters.gaussian and sigma is None:
        logger.warning("Please provide a sigma for the gaussian filter")
        return False
    if filter_type not in Filters._member_names_:
        logger.warning(f"Unknown filter type {filter_type}")
        return False
    return True


def is_positive(value: Optional[float]) -> bool:
    if value is None:
        return False
    return value > 0


def apply_filter(
    array: Array,
    filter_type: Filters,
    size: Optional[int] = None,
    sigma: Optional[float] = None,
) -> Array:
    if not valid_filter(filter_type=filter_type, size=size, sigma=sigma):
        return array

    is_numpy = False
    if isinstance(array, np.ndarray):
        is_numpy = True
        array = da.from_array(array)

    if filter_type == Filters.median and is_positive(size):
        array = dask_image.ndfilters.median_filter(array, size)

    elif filter_type == Filters.gaussian and is_positive(sigma):
        array = dask_image.ndfilters.gaussian_filter(array, sigma)

    if is_numpy:
        array = array.compute()

    return array


def check_frame_dimensions(frames, reference_image):
    if not isinstance(frames, np.ndarray):
        frames = np.asanyarray(frames)
    if len(frames.shape) != 3:
        raise ShapeError(f"Expected frame to be 3 dimensional, got {frames.shape}")
    if len(reference_image.shape) != 2:
        raise ShapeError(
            f"Expected refernce image to be 2 dimensional, got {frames.shape}",
        )
    num_frames = frames.shape[-1]
    shape = reference_image.shape
    if frames.shape != (shape[0], shape[1], num_frames):
        msg = (
            "Shape mistmact between frames and reference image. "
            f"Got frames with shape {frames.shape}, and reference "
            f"image with shape {reference_image.shape}"
        )
        raise ShapeError(msg)

    return frames


class MPSData:
    def __init__(self, frames, time_stamps, info, pacing=None, metadata=None) -> None:
        self.frames = frames
        self.time_stamps = time_stamps
        self.info = info
        if pacing is None:
            pacing = np.zeros_like(time_stamps)
        self.pacing = pacing
        if metadata is None:
            metadata = {}
        self.metadata = metadata

    @property
    def size_x(self):
        return self.frames.shape[0]

    @property
    def size_y(self):
        return self.frames.shape[1]

    @property
    def num_frames(self):
        return self.frames.shape[2]

    @property
    def framerate(self):
        return int(1000 / np.mean(np.diff(self.time_stamps)))


def to_uint8(img):
    return (256 * (img.astype(float) / max(img.max(), 1e-8))).astype(np.uint8)
