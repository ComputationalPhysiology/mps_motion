"""
Farnebäck, G. (2003, June). Two-frame motion estimation based on polynomial expansion. In Scandinavian conference on Image analysis (pp. 363-370). Springer, Berlin, Heidelberg.


"""
import logging
from typing import Optional

import cv2
import dask
import dask.array as da
import numpy as np

from .utils import to_uint8

logger = logging.getLogger(__name__)


def default_options():
    return dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def flow_map(args):
    reference_image, image, *remaining_args = args
    return flow(reference_image, image, *remaining_args)


def flow(
    image: np.ndarray,
    reference_image: np.ndarray,
    mask_flow: Optional[np.ndarray] = None,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
    flags: int = 0,
):
    if image.dtype != "uint8":
        image = to_uint8(image)
    if reference_image.dtype != "uint8":
        reference_image = to_uint8(reference_image)

    return cv2.calcOpticalFlowFarneback(
        reference_image,
        image,
        None,
        pyr_scale,
        levels,
        winsize,
        iterations,
        poly_n,
        poly_sigma,
        flags,
    )


def get_displacements(
    frames,
    reference_image: np.ndarray,
    mask_flow: Optional[np.ndarray] = None,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
    flags: int = 0,
):

    logger.info("Get displacements using Farneback's algorithm")

    all_flows = []
    for im in np.rollaxis(frames, 2):
        all_flows.append(
            dask.delayed(flow)(
                im,
                reference_image,
                mask_flow,
                pyr_scale,
                levels,
                winsize,
                iterations,
                poly_n,
                poly_sigma,
                flags,
            ),
        )
    flows = da.stack(*da.compute(all_flows), axis=-1)

    return flows
