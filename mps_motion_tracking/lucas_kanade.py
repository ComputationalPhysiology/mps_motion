"""
Lucas, B. D., & Kanade, T. (1981). An iterative image registration technique with an application to stereo vision.


http://cseweb.ucsd.edu/classes/sp02/cse252/lucaskanade81.pdf
"""
import concurrent.futures
from collections import namedtuple

import cv2
import numpy as np
import tqdm

from .utils import to_uint8

LKFlow = namedtuple("LKFlow", ["flow", "points"])


def default_options():
    pass


def flow_map(args):
    reference_image, image, *remaining_args = args

    return flow(to_uint8(reference_image), to_uint8(image), *remaining_args)


def flow(
    image: np.ndarray,
    reference_image: np.ndarray,
    points: np.ndarray,
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    return_points=False,
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


def get_displacements(
    frames,
    reference_image: np.ndarray,
    step=48,
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    return_refpoints=True,
):

    h, w = reference_image.shape[:2]
    grid = np.mgrid[step / 2 : w : step, step / 2 : h : step].astype(int)
    reference_points = np.expand_dims(grid.astype(np.float32).reshape(2, -1).T, 1)

    args = (
        (im, reference_image, reference_points, winSize, maxLevel, criteria, False)
        for im in np.rollaxis(frames, 2)
    )
    num_frames = frames.shape[-1]
    flows = np.zeros((reference_points.shape[0], 2, num_frames))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, uv in tqdm.tqdm(
            enumerate(executor.map(flow_map, args)), total=num_frames
        ):
            flows[:, :, i] = uv

    if return_refpoints:
        return LKFlow(flows, reference_points)
    return flows
