import logging

import cv2
import numpy as np

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


STEP = 48
QUIVER = (0, 0, 255)


def _draw_flow(image, x, y, fx, fy):
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
