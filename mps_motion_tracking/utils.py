import logging
from collections import namedtuple

import cv2
import numpy as np
import tqdm

logger = logging.getLogger(__name__)

try:
    from numba import jit, prange
except ImportError:
    msg = (
        "numba not found - Numba is just to speed up the motion tracking algorithm\n"
        "To install numba use: pip install numba"
    )
    logger.warning(msg)
    prange = range

    # Create a dummy decorator
    def jit(**params):
        def decorator(f):
            def wrap(*args, **kwargs):
                return f(*args, **kwargs)

            return wrap

        return decorator


_MPSData = namedtuple("MPSData", ["frames", "time_stamps", "info"])


class MPSData(_MPSData):
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


def resize_frames(frames: np.ndarray, scale: float = 1.0) -> np.ndarray:

    if scale < 1.0:

        w, h, num_frames = frames.shape

        width = int(w * scale)
        height = int(h * scale)

        resized_frames = np.zeros((width, height, num_frames))
        for i in tqdm.tqdm(range(num_frames)):
            resized_frames[:, :, i] = cv2.resize(frames[:, :, i], (height, width))
    else:
        resized_frames = frames.copy()

    return resized_frames


def interpolate_lk_flow(
    disp: np.ndarray,
    reference_points: np.ndarray,
    size_x: int,
    size_y: int,
    interpolation_method: str = "linear",
) -> np.ndarray:
    """Given an array of displacements (of flow) coming from
    the Lucas Kanade method return a new array which
    interpolates the data onto a given size, i.e
    the original size of the image.

    Parameters
    ----------
    disp : np.ndarray
        The flow or displacement from LK algorithm
    reference_points : np.ndarray
        Reference points
    size_x : int
        Size of the output in x-direction
    size_y : int
        Size of the output in y-direction
    interpolation_method : str
        Method for interpolation, by default 'linear'

    Returns
    -------
    np.ndarray
        Interpolated values
    """
    num_frames = disp.shape[-1]
    from scipy.interpolate import griddata

    disp_full = np.zeros((size_y, size_x, 2, num_frames))
    ref_points = np.squeeze(reference_points)
    grid_x, grid_y = np.meshgrid(np.arange(size_x), np.arange(size_y))
    # This could be parallelized
    for i in tqdm.tqdm(range(num_frames)):
        values_x = disp[:, 0, i]
        values_y = disp[:, 1, i]

        disp_full[:, :, 0, i] = griddata(
            ref_points, values_x, (grid_y, grid_x), method=interpolation_method
        )
        disp_full[:, :, 1, i] = griddata(
            ref_points, values_y, (grid_y, grid_x), method=interpolation_method
        )
    return disp_full


def _draw_flow(image, x, y, fx, fy):
    QUIVER = (0, 0, 255)
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, QUIVER, 5)
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_lk_flow(image, flow, reference_points):
    x, y = reference_points.reshape(-1, 2).astype(int).T
    fx, fy = flow.T
    return _draw_flow(image, x, y, fx, fy)


def draw_flow(image, flow, step=16):
    """[summary]

    Parameters
    ----------
    image : [type]
        [description]
    flow : [type]
        [description]
    step : int, optional
        [description], by default 16

    Returns
    -------
    [type]
        [description]
    """
    h, w = image.shape[:2]
    y, x = np.mgrid[step / 2 : h : step, step / 2 : w : step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    return _draw_flow(image, x, y, fx, fy)


def draw_hsv(flow):
    h, w = flow.shape[:2]
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((h, w, 3), np.uint8)
    # Sets image hue according to the optical flow
    # direction
    hsv[..., 0] = angle * 180 / np.pi / 2

    # Sets image saturation to maximum
    hsv[..., 1] = 255

    # Sets image value according to the optical flow
    # magnitude (normalized)
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return rgb


def to_uint8(img):
    return (img / 256).astype(np.uint8)
