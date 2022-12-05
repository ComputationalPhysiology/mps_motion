#!/usr/bin/env python3
# c) 2001-2021 Simula Research Laboratory ALL RIGHTS RESERVED
#
# END-USER LICENSE AGREEMENT
# PLEASE READ THIS DOCUMENT CAREFULLY. By installing or using this
# software you agree with the terms and conditions of this license
# agreement. If you do not accept the terms of this license agreement
# you may not install or use this software.
# Permission to use, copy, modify and distribute any part of this
# software for non-profit educational and research purposes, without
# fee, and without a written agreement is hereby granted, provided
# that the above copyright notice, and this license agreement in its
# entirety appear in all copies. Those desiring to use this software
# for commercial purposes should contact Simula Research Laboratory AS:
# post@simula.no
#
# IN NO EVENT SHALL SIMULA RESEARCH LABORATORY BE LIABLE TO ANY PARTY
# FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
# INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE
# "MPS" EVEN IF SIMULA RESEARCH LABORATORY HAS BEEN ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE. THE SOFTWARE PROVIDED HEREIN IS
# ON AN "AS IS" BASIS, AND SIMULA RESEARCH LABORATORY HAS NO OBLIGATION
# TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
# SIMULA RESEARCH LABORATORY MAKES NO REPRESENTATIONS AND EXTENDS NO
# WARRANTIES OF ANY KIND, EITHER IMPLIED OR EXPRESSED, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS
import concurrent.futures
import logging
from typing import Tuple
from typing import Union

import numpy as np
import tqdm

from . import scaling
from . import utils


__author__ = "Henrik Finsberg (henriknf@simula.no), 2017--2020"
__maintainer__ = "Henrik Finsberg"
__email__ = "henriknf@simula.no"


logger = logging.getLogger(__name__)


def default_options():
    return dict(block_size="auto", max_block_movement="auto")


def flow(
    image: np.ndarray,
    reference_image: np.ndarray,
    block_size: Union[str, int] = "auto",
    max_block_movement: Union[str, int] = "auto",
    resize: bool = True,
):
    """
    Computes the displacements from `reference_image` to `image`
    using a block matching algorithm. Briefly, we subdivde the images
    into blocks of size `block_size x block_size` and compare the two images
    within a range of +/- max_block_movement for each block.

    Arguments
    ---------
    image : np.ndarray
        The frame that you want to compute displacement for relative to
        the referernce frame
    reference_image : np.ndarray
        The frame used as reference
    block_size : int
        Size of the blocks
    max_block_movement : int
        Maximum allowed movement of blocks when searching for best match.
    resize: bool
        If True, make sure to resize the output images to have the same
        shape as the input, by default True.

    Note
    ----
    Make sure to have max_block_movement big enough. If this is too small
    then the results will be wrong. It is better to choose a too large value
    of this. However, choosing a too large value will mean that you need to
    compare more blocks which will increase the running time.
    """

    block_size = resolve_block_size(block_size, reference_image.shape)
    max_block_movement = resolve_max_block_movement(max_block_movement, block_size)

    vectors = _flow(reference_image, image, block_size, max_block_movement)

    if resize:
        new_shape: Tuple[int, int] = (
            reference_image.shape[0],
            reference_image.shape[1],
        )
        vectors = scaling.resize_vectors(vectors, new_shape)
    return vectors


@utils.jit(nopython=True)
def _flow(
    image: np.ndarray,
    reference_image: np.ndarray,
    block_size: int = 9,
    max_block_movement: int = 18,
):
    """
    Computes the displacements from `reference_image` to `image`
    using a block matching algorithm. Briefly, we subdivde the images
    into blocks of size `block_size x block_size` and compare the two images
    within a range of +/- max_block_movement for each block.

    Arguments
    ---------
    reference_image : np.ndarray
        The frame used as reference
    image : np.ndarray
        The frame that you want to compute displacement for relative to
        the referernce frame
    block_size : int
        Size of the blocks
    max_block_movement : int
        Maximum allowed movement of blocks when searching for best match.

    Note
    ----
    Make sure to have max_block_movement big enough. If this is too small
    then the results will be wrong. It is better to choose a too large value
    of this. However, choosing a too large value will mean that you need to
    compare more blocks which will increase the running time.
    """
    # Shape of the image that is returned
    y_size, x_size = image.shape
    block_size = max(block_size, 1)
    shape = (max(y_size // block_size, 1), max(x_size // block_size, 1))
    vectors = np.zeros((shape[0], shape[1], 2))
    costs = np.ones((2 * max_block_movement + 1, 2 * max_block_movement + 1))

    # Need to copy images to float array
    # otherwise negative values will be converted to large 16-bit integers
    ref_block = np.zeros((block_size, block_size))  # Block for reference image
    block = np.zeros((block_size, block_size))  # Block for image

    # Loop over each block
    for y_block in range(shape[0]):
        for x_block in range(shape[1]):

            # Coordinates in the orignal imagee
            y_image = y_block * block_size
            x_image = x_block * block_size

            block[:] = image[
                y_image : y_image + block_size,
                x_image : x_image + block_size,
            ]

            # Check if box has values
            if np.max(block) > 0:

                # Loop over values around the block within the `max_block_movement` range
                for i, y_block_ref in enumerate(
                    range(-max_block_movement, max_block_movement + 1),
                ):
                    for j, x_block_ref in enumerate(
                        range(
                            -max_block_movement,
                            max_block_movement + 1,
                        ),
                    ):

                        y_image_ref = y_image + y_block_ref
                        x_image_ref = x_image + x_block_ref

                        # Just make sure that we are within the referece image
                        if (
                            y_image_ref < 0
                            or y_image_ref + block_size > y_size
                            or x_image_ref < 0
                            or x_image_ref + block_size > x_size
                        ):
                            costs[i, j] = np.nan
                        else:
                            ref_block[:] = reference_image[
                                y_image_ref : y_image_ref + block_size,
                                x_image_ref : x_image_ref + block_size,
                            ]
                            # Could improve this cost function / template matching
                            costs[i, j] = np.sum(
                                np.abs(np.subtract(block, ref_block)),
                            ) / (block_size**2)

                # Find minimum cost vector and store it
                dy, dx = np.where(costs == np.nanmin(costs))

                # If there are more then one minima then we select none
                if len(dy) > 1 or len(dx) > 1:
                    vectors[y_block, x_block, 0] = 0
                    vectors[y_block, x_block, 1] = 0
                else:
                    vectors[y_block, x_block, 0] = dx[0] - max_block_movement
                    vectors[y_block, x_block, 1] = dy[0] - max_block_movement
            else:
                # If no values in box set to no movement
                vectors[y_block, x_block, :] = 0

    return vectors


def flow_map(args):
    """
    Helper function for running block maching algorithm in paralell
    """
    return _flow(*args)


def resolve_block_size(block_size: Union[str, int], shape: Tuple[int, ...]) -> int:
    if isinstance(block_size, str):  # == "auto":
        block_size = max(int(min(shape) / 128), 2)
    return block_size


def resolve_max_block_movement(
    max_block_movement: Union[str, int],
    block_size: int,
) -> int:
    if isinstance(max_block_movement, str):  # == "auto":
        max_block_movement = 2 * block_size
    return max_block_movement


def get_displacements(
    frames: np.ndarray,
    reference_image: np.ndarray,
    block_size: Union[str, int] = "auto",
    max_block_movement: Union[str, int] = "auto",
    resize=True,
    **kwargs,
) -> utils.Array:
    """Computes the displacements from `reference_image` to all `frames`
    using a block matching algorithm. Briefly, we subdivde the images
    into blocks of size `block_size x block_size` and compare the two images
    within a range of +/- max_block_movement for each block.

    Arguments
    ---------
    frames : np.ndarray
        The frame that you want to compute displacement for relative to
        the referernce frame. Input must be of shape (N, M, T, 2), where
        N is the width, M is the height and  T is the number
        number of frames.
    reference_image : np.ndarray
        The frame used as reference
    block_size : int
        Size of the blocks, by default 9
    max_block_movement : int
        Maximum allowed movement of blocks when searching for best match,
        by default 18.
    resize: bool
        If True, make sure to resize the output images to have the same
        shape as the input, by default True.

    Note
    ----
    Make sure to have max_block_movement big enough. If this is too small
    then the results will be wrong. It is better to choose a too large value
    of this. However, choosing a too large value will mean that you need to
    compare more blocks which will increase the running time.

    Returns
    -------
    utils.Array
        utils.Array with displacement of shape (N', M', T', 2), where
        N' is the width, M' is the height and  T' is the number
        number of frames. Note if `resize=True` then we have
        (N, M, T, 2) = (N', M', T', 2).
    """
    if kwargs:
        logger.warning(f"Unknown arguments {kwargs!r} - ignoring")
    frames = utils.check_frame_dimensions(frames, reference_image)

    block_size = resolve_block_size(block_size, reference_image.shape)
    max_block_movement = resolve_max_block_movement(max_block_movement, block_size)

    logger.info("Get displacements using block mathching")
    args = (
        (im, reference_image, block_size, max_block_movement)
        for im in np.rollaxis(frames, 2)
    )
    num_frames = frames.shape[-1]

    y_size, x_size = reference_image.shape
    block_size = max(block_size, 1)
    shape = (max(y_size // block_size, 1), max(x_size // block_size, 1))
    flows = np.zeros((shape[0], shape[1], num_frames, 2))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, uv in tqdm.tqdm(
            enumerate(executor.map(flow_map, args)),
            desc="Compute displacement",
            total=num_frames,
        ):
            flows[:, :, i, :] = uv

    if resize:

        new_shape: Tuple[int, int] = (
            reference_image.shape[0],
            reference_image.shape[1],
        )
        flows = scaling.resize_vectors(flows, new_shape)

    return flows
