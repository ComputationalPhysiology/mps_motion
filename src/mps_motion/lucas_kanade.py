"""
Lucas, B. D., & Kanade, T. (1981). An iterative image registration technique with an application to stereo vision.


http://cseweb.ucsd.edu/classes/sp02/cse252/lucaskanade81.pdf
"""
import concurrent.futures
import logging
from enum import Enum
from typing import Optional
from typing import Tuple
from typing import Union

import cv2
import dask
import dask.array as da
import numpy as np
import tqdm
from dask.diagnostics import ProgressBar

from . import scaling
from . import utils

logger = logging.getLogger(__name__)


class Interpolation(str, Enum):
    none = "none"
    reshape = "reshape"
    rbf = "rbf"
    nearest = "nearest"


def default_options():
    return dict(
        winSize=(15, 15),
        maxLevel=2,
        interpolation=Interpolation.nearest,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        step="auto",
    )


def resolve_step(step: Union[str, int], shape: Tuple[int, ...]) -> int:
    if isinstance(step, str):  # == "auto":
        step = max(int(min(shape) / 24), 1)

    return step


def flow(
    image: np.ndarray,
    reference_image: np.ndarray,
    points: Optional[np.ndarray] = None,
    winSize: Tuple[int, int] = (15, 15),
    maxLevel: int = 2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    step: Union[str, int] = "auto",
    interpolation: Interpolation = Interpolation.nearest,
) -> np.ndarray:
    """Compute the optical from reference_image to image

    Parameters
    ----------
    image : np.ndarray
        The target image
    reference_image : np.ndarray
        The reference image
    points : np.ndarray
        Points where to compute the motion vectors
    winSize : Tuple[int, int], optional
        Size of search window in each pyramid level, by default (15, 15)
    maxLevel : int, optional
        0-based maximal pyramid level number; if set to 0,
        pyramids are not used (single level), if set to 1,
        two levels are used, and so on; if pyramids are
        passed to input then algorithm will use as many
        levels as pyramids have but no more than maxLevel, by default 2
    criteria : tuple, optional
        Parameter, specifying the termination criteria of the iterative
        search algorithm (after the specified maximum number of iterations
        criteria.maxCount or when the search window moves by less than
        criteria.epsilon, by default (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    step : int, optional
        Step in pixels between points where motion is computed, by default 16
    interpolation : Interpolation
        Interpolate flow to original shape using radial basis function ('rbf'),
        nearest neigbour interpolation ('nearest') or do not interpolate but reshape ('reshape'),
        or use the original output from the LK algorithm ('none'), by default 'nearest'

    Returns
    -------
    np.ndarray
        Array of optical flow from reference image to image
    """
    if points is None:
        step = resolve_step(step, reference_image.shape)
        points = get_uniform_reference_points(reference_image, step=step)
    if image.dtype != np.uint8:
        image = utils.to_uint8(image)
    if reference_image.dtype != np.uint8:
        reference_image = utils.to_uint8(reference_image)

    f = _flow(image, reference_image, points, winSize, maxLevel, criteria)
    points = points.squeeze()

    if interpolation == Interpolation.none:
        return f

    if interpolation == Interpolation.rbf:
        return scaling.rbfinterp2d(
            points,
            f,
            np.arange(image.shape[1]),
            np.arange(image.shape[0]),
        )

    f = scaling.reshape_lk(points, f)
    if interpolation == Interpolation.nearest:
        new_shape: Tuple[int, int] = (
            reference_image.shape[0],
            reference_image.shape[1],
        )
        f = scaling.resize_vectors(f, new_shape)
    return f


def _flow(
    image: np.ndarray,
    reference_image: np.ndarray,
    points: np.ndarray,
    winSize: Tuple[int, int] = (15, 15),
    maxLevel: int = 2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
) -> np.ndarray:
    """Compute the optical from reference_image to image

    Parameters
    ----------
    image : np.ndarray
        The target image
    reference_image : np.ndarray
        The reference image
    points : np.ndarray
        Points where to compute the motion vectors
    winSize : Tuple[int, int], optional
        Size of search window in each pyramid level, by default (15, 15)
    maxLevel : int, optional
        0-based maximal pyramid level number; if set to 0,
        pyramids are not used (single level), if set to 1,
        two levels are used, and so on; if pyramids are
        passed to input then algorithm will use as many
        levels as pyramids have but no more than maxLevel, by default 2
    criteria : tuple, optional
        Parameter, specifying the termination criteria of the iterative
        search algorithm (after the specified maximum number of iterations
        criteria.maxCount or when the search window moves by less than
        criteria.epsilon, by default (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

    Returns
    -------
    np.ndarray
        Array of optical flow from reference image to image
    """

    if image.dtype != np.uint8:
        image = utils.to_uint8(image)
    if reference_image.dtype != np.uint8:
        reference_image = utils.to_uint8(reference_image)

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


def get_uniform_reference_points(image: np.ndarray, step: int = 48) -> np.ndarray:
    """Create a grid of uniformly spaced points width
    the gived steps size constraind by the image
    dimensions.

    Parameters
    ----------
    image : np.ndarray
        An image used to set the constraint on the dimensions
    step : int, optional
        The stepsize, by default 48

    Returns
    -------
    np.ndarray
        An array of uniformly spaced points
    """
    h, w = image.shape[:2]
    grid = np.mgrid[step // 2 : w : step, step // 2 : h : step].astype(int)
    return np.expand_dims(grid.astype(np.float32).reshape(2, -1).T, 1)


def get_displacements(
    frames,
    reference_image: np.ndarray,
    step: Union[str, int] = "auto",
    winSize: Tuple[int, int] = (15, 15),
    maxLevel: int = 2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    interpolation: Interpolation = Interpolation.nearest,
    **kwargs,
) -> np.ndarray:
    """Compute the optical flow using the Lucas Kanade method from
    the reference frame to all other frames

    Parameters
    ----------
    frames : np.ndarray
        The frames with some moving objects. Input must be of shape (N, M, T, 2), where
        N is the width, M is the height and  T is the number
    reference_image : np.ndarray
        The reference image
    step : int, optional
        Step in pixels between points where motion is computed, by default 16
    winSize : Tuple[int, int], optional
        Size of search window in each pyramid level, by default (15, 15)
    maxLevel : int, optional
        0-based maximal pyramid level number; if set to 0,
        pyramids are not used (single level), if set to 1,
        two levels are used, and so on; if pyramids are
        passed to input then algorithm will use as many
        levels as pyramids have but no more than maxLevel, by default 2
    criteria : tuple, optional
        Parameter, specifying the termination criteria of the iterative
        search algorithm (after the specified maximum number of iterations
        criteria.maxCount or when the search window moves by less than
        criteria.epsilon, by default (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    interpolation : Interpolation
        Interpolate flow to original shape using radial basis function ('rbf'),
        nearest neigbour interpolation ('nearest') or do not interpolate but reshape ('reshape'),
        or use the original output from the LK algorithm ('none'), by default 'nearest'

    Returns
    -------
    Array
        An array of motion vectors relative to the reference image. If shape of
        input frames are (N, M, T) then the shape of the output is (N', M', T', 2).
        Note if `resize=True` then we have (N, M, T, 2) = (N', M', T', 2).
    """
    if kwargs:
        logger.warning(f"Unknown arguments {kwargs!r} - ignoring")
    logger.info("Get displacements using Lucas Kanade")

    frames = utils.check_frame_dimensions(frames, reference_image)

    step = resolve_step(step, reference_image.shape)
    reference_points = get_uniform_reference_points(reference_image, step=step)

    num_frames = frames.shape[-1]

    all_flows = []
    for im in np.rollaxis(frames, 2):
        all_flows.append(
            dask.delayed(_flow)(
                im,
                reference_image,
                reference_points,
                winSize,
                maxLevel,
                criteria,
            ),
        )

    with ProgressBar(out=utils.LoggerWrapper(logger, "info")):
        flows = da.stack(*da.compute(all_flows), axis=2)
    logger.info("Done with Lucas-Kanade method")

    if interpolation == Interpolation.none:
        return flows

    if interpolation == Interpolation.rbf:
        int_flows = np.zeros(
            (reference_image.shape[0], reference_image.shape[1], 2, num_frames),
        )
        p = reference_points.squeeze()
        x = np.arange(reference_image.shape[1])
        y = np.arange(reference_image.shape[0])
        int_args = ((p, f, x, y) for f in np.rollaxis(flows, 2))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for i, q in tqdm.tqdm(
                enumerate(executor.map(scaling.rbfinterp2d_map, int_args)),
                desc="Interpolate",
                total=num_frames,
            ):
                int_flows[:, :, :, i] = q
        return int_flows

    flows = scaling.reshape_lk(reference_points, flows)

    if interpolation == Interpolation.nearest:

        new_shape: Tuple[int, int] = (
            reference_image.shape[0],
            reference_image.shape[1],
        )

        flows = scaling.resize_vectors(flows, new_shape)
    return flows
