import logging
import os
from typing import Union

import dask.array as da
import numpy as np


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
            logger.warning(
                "You are trying to call a numba function, but numba is not installed",
            )

            def wrap(*args, **kwargs):
                return f(*args, **kwargs)

            return wrap

        return decorator


class ShapeError(RuntimeError):
    pass


def download_demo_data(path):
    print("Downloading data. Please wait...")
    link = "https://www.dropbox.com/s/xbn29petfkpjf7w/PointH4A_ChannelBF_VC_Seq0018.nd2?dl=1"
    import urllib.request
    import time

    urllib.request.urlretrieve(link, path)
    time.sleep(1.0)
    print("Done downloading data")


def unmask(array, fill_value=0.0):
    """Take a mask array and fills
    it with the given fill value and
    return an regular numpy array
    """
    if isinstance(array, np.ma.MaskedArray):
        return array.filled(fill_value)
    return array


PathLike = Union[str, os.PathLike]
Array = Union[da.core.Array, np.ndarray]


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
    img_float = img.astype(float)
    return (256 * (img_float / max(img_float.max(), 1e-12))).astype(np.uint8)


def ca_transient(
    t: np.ndarray,
    tstart: float = 0.05,
    tau1: float = 0.05,
    tau2: float = 0.110,
    ca_diast: float = 0.0,
    ca_ampl: float = 1.0,
) -> np.ndarray:

    beta = (tau1 / tau2) ** (-1 / (tau1 / tau2 - 1)) - (tau1 / tau2) ** (
        -1 / (1 - tau2 / tau1)
    )
    ca = np.zeros_like(t)

    ca[t <= tstart] = ca_diast

    ca[t > tstart] = (ca_ampl - ca_diast) / beta * (
        np.exp(-(t[t > tstart] - tstart) / tau1)
        - np.exp(-(t[t > tstart] - tstart) / tau2)
    ) + ca_diast
    return ca
