"""
Farnebäck, G. (2003, June). Two-frame motion estimation based on polynomial expansion. In Scandinavian conference on Image analysis (pp. 363-370). Springer, Berlin, Heidelberg.
"""
import logging

import cv2
import dask
import dask.array as da
import numpy as np
from dask.diagnostics import ProgressBar

from . import utils

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


def flow(
    image: np.ndarray,
    reference_image: np.ndarray,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
    flags: int = 0,
    factor: float = 1.0,
):
    """
    Compute the optical flow using the Farneback method from
    the reference frame to another image

    Parameters
    ----------
    image : np.ndarray
        The target image
    reference_image : np.ndarray
        The reference image
    pyr_scale : float, optional
        parameter, specifying the image scale (<1) to build pyramids
        for each image; pyr_scale=0.5 means a classical pyramid,
        where each next layer is twice smaller than the previous
        one, by default 0.5
    levels : int, optional
        number of pyramid layers including the initial image; levels=1
        means that no extra layers are created and only the original
        images are used, by default 3
    winsize : int, optional
        averaging window size; larger values increase the algorithm
        robustness to image noise and give more chances for fast motion
        detection, but yield more blurred motion field, by default 15
    iterations : int, optional
        number of iterations the algorithm does at each pyramid level, by default 3
    poly_n : int, optional
        size of the pixel neighborhood used to find polynomial expansion in each pixel.
        larger values mean that the image will be approximated with smoother surfaces,
        yielding more robust algorithm and more blurred motion field,
        typically poly_n =5 or 7., by default 5
    poly_sigma : float, optional
        standard deviation of the Gaussian that is used to smooth derivatives used as a
        basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1,
        for poly_n=7, a good value would be poly_sigma=1.5, by default 1.2
    flags : int, optional
        By default 0. operation flags that can be a combination of the following:
        - OPTFLOW_USE_INITIAL_FLOW uses the input flow as an initial flow approximation.
        - OPTFLOW_FARNEBACK_GAUSSIAN uses the Gaussian winsize x winsize filter
        instead of a box filter of the same size for optical flow estimation;
        usually, this option gives z more accurate flow than with a box filter,
        at the cost of lower speed; normally, winsize for a Gaussian window should
        be set to a larger value to achieve the same level of robustness.
    factor: float
        Factor to multiply the result

    Returns
    -------
    np.ndarray
        The motion vectors
    """
    if image.dtype != "uint8":
        image = utils.to_uint8(image)
    if reference_image.dtype != "uint8":
        reference_image = utils.to_uint8(reference_image)

    return factor * cv2.calcOpticalFlowFarneback(
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
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
    flags: int = 0,
    **kwargs,
) -> utils.Array:
    """Compute the optical flow using the Farneback method from
    the reference frame to all other frames

    Parameters
    ----------
    frames : np.ndarray
        The frames with some moving objects
    reference_image : np.ndarray
        The reference image
    pyr_scale : float, optional
        parameter, specifying the image scale (<1) to build pyramids
        for each image; pyr_scale=0.5 means a classical pyramid,
        where each next layer is twice smaller than the previous
        one, by default 0.5
    levels : int, optional
        number of pyramid layers including the initial image; levels=1
        means that no extra layers are created and only the original
        images are used, by default 3
    winsize : int, optional
        averaging window size; larger values increase the algorithm
        robustness to image noise and give more chances for fast motion
        detection, but yield more blurred motion field, by default 15
    iterations : int, optional
        number of iterations the algorithm does at each pyramid level, by default 3
    poly_n : int, optional
        size of the pixel neighborhood used to find polynomial expansion in each pixel.
        larger values mean that the image will be approximated with smoother surfaces,
        yielding more robust algorithm and more blurred motion field,
        typically poly_n =5 or 7., by default 5
    poly_sigma : float, optional
        standard deviation of the Gaussian that is used to smooth derivatives used as a
        basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1,
        for poly_n=7, a good value would be poly_sigma=1.5, by default 1.2
    flags : int, optional
        By default 0. operation flags that can be a combination of the following:
        - OPTFLOW_USE_INITIAL_FLOW uses the input flow as an initial flow approximation.
        - OPTFLOW_FARNEBACK_GAUSSIAN uses the Gaussian winsize x winsize filter
        instead of a box filter of the same size for optical flow estimation;
        usually, this option gives z more accurate flow than with a box filter,
        at the cost of lower speed; normally, winsize for a Gaussian window should
        be set to a larger value to achieve the same level of robustness.

    Returns
    -------
    Array
        An array of motion vectors relative to the reference image. If shape of
        input frames are (N, M, T) then the shape of the output is (N, M, T, 2).
    """
    if kwargs:
        logger.warning(f"Unknown arguments {kwargs!r} - ignoring")
    logger.info("Get displacements using Farneback's algorithm")

    all_flows = []
    for im in np.rollaxis(frames, 2):
        all_flows.append(
            dask.delayed(flow)(
                im,
                reference_image,
                pyr_scale,
                levels,
                winsize,
                iterations,
                poly_n,
                poly_sigma,
                flags,
            ),
        )

    with ProgressBar():
        arr = da.compute(all_flows)
    flows = np.stack(*arr, axis=2)  # type:ignore

    logger.info("Done running Farneback's algorithm")
    return da.from_array(flows)


def get_velocities(
    frames: np.ndarray,
    time_stamps: np.ndarray,
    spacing: int = 1,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
    flags: int = 0,
) -> utils.Array:
    """Compute the optical flow using the Farneback method from
    the reference frame to all other frames

    Parameters
    ----------
    frames : np.ndarray
        The frames with some moving objects
    time_stamps : np.ndarray
        Time stamps
    spacing : int
        Spacing between frames used to compute velocities
    pyr_scale : float, optional
        parameter, specifying the image scale (<1) to build pyramids
        for each image; pyr_scale=0.5 means a classical pyramid,
        where each next layer is twice smaller than the previous
        one, by default 0.5
    levels : int, optional
        number of pyramid layers including the initial image; levels=1
        means that no extra layers are created and only the original
        images are used, by default 3
    winsize : int, optional
        averaging window size; larger values increase the algorithm
        robustness to image noise and give more chances for fast motion
        detection, but yield more blurred motion field, by default 15
    iterations : int, optional
        number of iterations the algorithm does at each pyramid level, by default 3
    poly_n : int, optional
        size of the pixel neighborhood used to find polynomial expansion in each pixel.
        larger values mean that the image will be approximated with smoother surfaces,
        yielding more robust algorithm and more blurred motion field,
        typically poly_n =5 or 7., by default 5
    poly_sigma : float, optional
        standard deviation of the Gaussian that is used to smooth derivatives used as a
        basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1,
        for poly_n=7, a good value would be poly_sigma=1.5, by default 1.2
    flags : int, optional
        By default 0. operation flags that can be a combination of the following:
        - OPTFLOW_USE_INITIAL_FLOW uses the input flow as an initial flow approximation.
        - OPTFLOW_FARNEBACK_GAUSSIAN uses the Gaussian winsize x winsize filter
        instead of a box filter of the same size for optical flow estimation;
        usually, this option gives z more accurate flow than with a box filter,
        at the cost of lower speed; normally, winsize for a Gaussian window should
        be set to a larger value to achieve the same level of robustness.

    Returns
    -------
    Array
        An array of motion vectors relative to the reference image. If shape of
        input frames are (N, M, T) then the shape of the output is (N, M, T, 2).
    """

    logger.info("Get velocities using Farneback's algorithm")
    dts = np.subtract(time_stamps[spacing:], time_stamps[:-spacing])
    all_flows = []
    for index in range(frames.shape[-1] - spacing):
        all_flows.append(
            dask.delayed(flow)(
                frames[:, :, index + spacing],
                frames[:, :, index],
                pyr_scale,
                levels,
                winsize,
                iterations,
                poly_n,
                poly_sigma,
                flags,
                (1.0 / dts[index]),
            ),
        )

    with ProgressBar():
        arr = da.compute(all_flows)

    flows = np.stack(*arr, axis=2)  # type:ignore

    logger.info("Done running Farneback's algorithm")

    return da.from_array(flows)
