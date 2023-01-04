"""

ZACH, Christopher; POCK, Thomas; BISCHOF, Horst. A duality based approach for realtime tv-l 1 optical flow. In: Joint pattern recognition symposium. Springer, Berlin, Heidelberg, 2007. p. 214-223.

http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.709.4597&rep=rep1&type=pdf

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
    return {"tau": 0.25, "lmbda": 0.08, "theta": 0.37, "nscales": 6, "warps": 5}


def flow(
    image: np.ndarray,
    reference_image: np.ndarray,
    tau: float = 0.25,
    lmbda: float = 0.08,
    theta: float = 0.37,
    nscales: int = 6,
    warps: int = 5,
) -> np.ndarray:

    dual_proc = cv2.optflow.DualTVL1OpticalFlow_create(
        tau,
        lmbda,
        theta,
        nscales,
        warps,
    )
    est_flow = np.zeros(
        shape=(reference_image.shape[0], reference_image.shape[1], 2),
        dtype=np.float32,
    )

    if image.dtype != "uint8":
        image = utils.to_uint8(image)
    if reference_image.dtype != "uint8":
        reference_image = utils.to_uint8(reference_image)

    dual_proc.calc(
        reference_image,
        image,
        est_flow,
    )
    return est_flow


def get_displacements(
    frames: np.ndarray,
    reference_image: np.ndarray,
    tau: float = 0.25,
    lmbda: float = 0.08,
    theta: float = 0.37,
    nscales: int = 6,
    warps: int = 5,
    **kwargs,
) -> utils.Array:
    """Compute the optical flow using the Dual TV-L1 method from
    the reference frame to all other frames


    Parameters
    ----------
    frames : np.ndarray
        The frames with some moving objects
    reference_image : np.ndarray
        The reference image
    tau : float, optional
        Time step of the numerical scheme, by default 0.25
    lmbda : float, optional
        Weight parameter for the data term, attachment parameter.
        This is the most relevant parameter, which determines the smoothness
        of the output. The smaller this parameter is, the smoother the solutions
        we obtain. It depends on the range of motions of the images, so its
        value should be adapted to each image sequence, by default 0.08
    theta : float, optional
        parameter used for motion estimation. It adds a variable allowing for
        illumination variations Set this parameter to 1. if you have varying
        illumination. See: Chambolle et al, A First-Order Primal-Dual Algorithm
        for Convex Problems with Applications to Imaging Journal of Mathematical
        imaging and vision, may 2011 Vol 40 issue 1, pp 120-145, by default 0.37
    nscales : int, optional
        Number of scales used to create the pyramid of images, by default 6
    warps : int, optional
        Number of warpings per scale. Represents the number of times that I1(x+u0)
        and grad( I1(x+u0) ) are computed per scale. This is a parameter that assures
        the stability of the method. It also affects the running time, so it is a
        compromise between speed and accuracy., by default 5

    Returns
    -------
    Array
        An array of motion vectors relative to the reference image. If shape of
        input frames are (N, M, T) then the shape of the output is (N, M, T, 2).
    """
    if kwargs:
        logger.warning(f"Unknown arguments {kwargs!r} - ignoring")
    logger.info("Get displacements using Dual TV-L 1")

    all_flows = []
    for im in np.rollaxis(frames, 2):
        all_flows.append(
            dask.delayed(flow)(im, reference_image, tau, lmbda, theta, nscales, warps),
        )

    with ProgressBar(out=utils.LoggerWrapper(logger, "info")):
        arr = da.compute(all_flows)
    flows = np.stack(*arr, axis=2)  # type:ignore

    logger.info("Done running Dual TV-L 1 algorithm")

    return da.from_array(flows)
