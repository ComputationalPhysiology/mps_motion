import logging
import os
from typing import Union

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
            def wrap(*args, **kwargs):
                return f(*args, **kwargs)

            return wrap

        return decorator


class ShapeError(RuntimeError):
    pass


PathLike = Union[str, os.PathLike]


def check_frame_dimensions(frames, reference_image):
    if not isinstance(frames, np.ndarray):
        frames = np.asanyarray(frames)
    if len(frames.shape) != 3:
        raise ShapeError(f"Expected frame to be 3 dimensional, got {frames.shape}")
    if len(reference_image.shape) != 2:
        raise ShapeError(
            f"Expected refernce image to be 2 dimensional, got {frames.shape}"
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
    return (img / 256).astype(np.uint8)
