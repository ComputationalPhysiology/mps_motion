"""
Farnebäck, G. (2003, June). Two-frame motion estimation based on polynomial expansion. In Scandinavian conference on Image analysis (pp. 363-370). Springer, Berlin, Heidelberg.


"""

import concurrent.futures
import logging
from typing import Optional

import cv2
import numpy as np
import tqdm

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

    return flow(to_uint8(reference_image), to_uint8(image), *remaining_args)


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


def get_velocities(
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

    args = (
        (
            im,
            ref,
            mask_flow,
            pyr_scale,
            levels,
            winsize,
            iterations,
            poly_n,
            poly_sigma,
            flags,
        )
        for (im, ref) in zip(np.rollaxis(frames, 2)[1:], np.rollaxis(frames, 2)[:-1])
    )
    num_frames = frames.shape[-1]
    flows = np.zeros(
        (reference_image.shape[0], reference_image.shape[1], 2, num_frames)
    )
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, uv in tqdm.tqdm(
            enumerate(executor.map(flow_map, args)), total=num_frames
        ):
            flows[:, :, :, i] = uv

    return flows


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

    args = (
        (
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
        )
        for im in np.rollaxis(frames, 2)
    )
    num_frames = frames.shape[-1]
    flows = np.zeros(
        (reference_image.shape[0], reference_image.shape[1], 2, num_frames)
    )
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, uv in tqdm.tqdm(
            enumerate(executor.map(flow_map, args)), total=num_frames
        ):
            flows[:, :, :, i] = uv

    return flows
