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

from .utils import to_uint8

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
):

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
        image = to_uint8(image)
    if reference_image.dtype != "uint8":
        reference_image = to_uint8(reference_image)

    dual_proc.calc(
        reference_image,
        image,
        est_flow,
    )
    return est_flow


def get_displacements(
    frames,
    reference_image: np.ndarray,
    tau: float = 0.25,
    lmbda: float = 0.08,
    theta: float = 0.37,
    nscales: int = 6,
    warps: int = 5,
):

    logger.info("Get displacements using Dualt TV-L 1")

    all_flows = []
    for im in np.rollaxis(frames, 2):
        all_flows.append(
            dask.delayed(flow)(im, reference_image, tau, lmbda, theta, nscales, warps),
        )

    with ProgressBar():
        flows = da.stack(*da.compute(all_flows), axis=2)

    logger.info("Done running Dualt TV-L 1 algorithm")

    return flows
