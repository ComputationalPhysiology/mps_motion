"""
Lucas, B. D., & Kanade, T. (1981). An iterative image registration technique with an application to stereo vision.


http://cseweb.ucsd.edu/classes/sp02/cse252/lucaskanade81.pdf
"""
import concurrent.futures
import logging
from collections import namedtuple
from typing import Optional, Tuple

import cv2
import numpy as np
import tqdm

from . import utils

logger = logging.getLogger(__name__)
LKFlow = namedtuple("LKFlow", ["flow", "points"])


def default_options():
    return dict(
        winSize=(15, 15),
        maxLevel=2,
        interpolate=False,
        reshape=True,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        step=16,
    )


def rbfinterp2d_map(args):
    return utils.rbfinterp2d(*args)


def flow_map(args):
    reference_image, image, *remaining_args = args

    return _flow(
        utils.to_uint8(reference_image), utils.to_uint8(image), *remaining_args
    )


def flow(
    image: np.ndarray,
    reference_image: np.ndarray,
    points: Optional[np.ndarray] = None,
    winSize: Tuple[int, int] = (15, 15),
    maxLevel: int = 2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    return_points: bool = False,
    step: int = 16,
    interpolate=True,
) -> np.ndarray:
    if points is None:
        points = get_uniform_reference_points(reference_image, step=step)
    if image.dtype != np.uint8:
        image = utils.to_uint8(image)
    if reference_image.dtype != np.uint8:
        reference_image = utils.to_uint8(reference_image)

    f = _flow(image, reference_image, points, winSize, maxLevel, criteria)
    if interpolate:
        f = utils.rbfinterp2d(
            points.squeeze(), f, np.arange(image.shape[1]), np.arange(image.shape[0])
        )
    return f


def _flow(
    image: np.ndarray,
    reference_image: np.ndarray,
    points: np.ndarray,
    winSize: Tuple[int, int] = (15, 15),
    maxLevel: int = 2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
) -> np.ndarray:

    next_points, status, error = cv2.calcOpticalFlowPyrLK(
        reference_image,
        image,
        points,
        None,
        winSize=winSize,
        maxLevel=maxLevel,
        criteria=criteria,
    )

    flow = (next_points - points).reshape(-1, 2)

    return flow


def get_uniform_reference_points(image, step=48):
    h, w = image.shape[:2]
    grid = np.mgrid[step / 2 : w : step, step / 2 : h : step].astype(int)
    return np.expand_dims(grid.astype(np.float32).reshape(2, -1).T, 1)


def get_displacements(
    frames,
    reference_image: np.ndarray,
    step: int = 16,
    winSize: Tuple[int, int] = (15, 15),
    maxLevel: int = 2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    return_refpoints: bool = False,
    interpolate: bool = False,
    reshape: bool = True,
    resize: bool = True,
):
    logger.info("Get displacements using Lucas Kanade")

    reference_points = get_uniform_reference_points(reference_image, step=step)

    args = (
        (im, reference_image, reference_points, winSize, maxLevel, criteria)
        for im in np.rollaxis(frames, 2)
    )
    num_frames = frames.shape[-1]
    flows = np.zeros((reference_points.shape[0], 2, num_frames))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, uv in tqdm.tqdm(
            enumerate(executor.map(flow_map, args)),
            desc="Compute displacement",
            total=num_frames,
        ):
            flows[:, :, i] = uv

    if interpolate:
        int_flows = np.zeros(
            (reference_image.shape[0], reference_image.shape[1], 2, num_frames)
        )
        p = reference_points.squeeze()
        x = np.arange(reference_image.shape[1])
        y = np.arange(reference_image.shape[0])
        int_args = ((p, f, x, y) for f in np.rollaxis(flows, 2))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for i, q in tqdm.tqdm(
                enumerate(executor.map(rbfinterp2d_map, int_args)),
                desc="Interpolate",
                total=num_frames,
            ):
                int_flows[:, :, :, i] = q
        flows = int_flows
    else:
        if reshape:
            out = utils.reshape_lk(reference_points, flows)
            flows = out
        if resize:
            new_shape: Tuple[int, int] = (
                reference_image.shape[0],
                reference_image.shape[1],
            )
            int_flows = np.zeros((new_shape[0], new_shape[1], 2, num_frames))
            int_flows[:, :, 0, :] = utils.resize_frames(
                flows[:, :, 0, :], new_shape=new_shape
            )
            int_flows[:, :, 1, :] = utils.resize_frames(
                flows[:, :, 1, :], new_shape=new_shape
            )
            flows = int_flows

    if return_refpoints:
        return LKFlow(flows, reference_points)
    return flows
