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

import numpy as np
import tqdm

from .utils import jit, resize_frames

__author__ = "Henrik Finsberg (henriknf@simula.no), 2017--2020"
__maintainer__ = "Henrik Finsberg"
__email__ = "henriknf@simula.no"

"""

This module is based on the Matlab code in the script scripttotry1105_nd2.m
provided by Berenice

"""
logger = logging.getLogger(__name__)


def default_options():
    return dict(block_size=9, max_block_movement=18, filter_kernel_size=8)


def filter_vectors(vectors, filter_kernel_size):
    from scipy import ndimage

    vectors[:, :, 0] = ndimage.median_filter(vectors[:, :, 0], filter_kernel_size)
    vectors[:, :, 1] = ndimage.median_filter(vectors[:, :, 1], filter_kernel_size)
    return vectors


def flow_map(args):
    """

    Helper function for running block maching algorithm in paralell

    """
    return flow(*args)


def flow(
    reference_image: np.ndarray,
    image: np.ndarray,
    block_size: int = 9,
    max_block_movement: int = 18,
    filter_kernel_size: int = 5,
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
    vectors = _flow(reference_image, image, block_size, max_block_movement)

    if filter_kernel_size > 0:
        vectors = filter_vectors(vectors, filter_kernel_size)

    return vectors


@jit(nopython=True)
def _flow(
    reference_image: np.ndarray,
    image: np.ndarray,
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
    shape = (y_size // block_size, x_size // block_size)
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
                    range(-max_block_movement, max_block_movement + 1)
                ):
                    for j, x_block_ref in enumerate(
                        range(
                            -max_block_movement,
                            max_block_movement + 1,
                        )
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
                                np.abs(np.subtract(block, ref_block))
                            ) / (block_size ** 2)

                # Find minimum cost vector and store it
                dy, dx = np.where(costs == np.nanmin(costs))

                # If there are more then one minima then we select none
                if len(dy) > 1 or len(dx) > 1:
                    vectors[y_block, x_block, 0] = 0
                    vectors[y_block, x_block, 1] = 0
                else:
                    vectors[y_block, x_block, 0] = max_block_movement - dy[0]
                    vectors[y_block, x_block, 1] = max_block_movement - dx[0]
            else:
                # If no values in box set to no movement
                vectors[y_block, x_block, :] = 0

    return vectors


def get_displacements(
    frames,
    reference_image: np.ndarray,
    block_size: int = 9,
    max_block_movement: int = 18,
    filter_kernel_size: int = 9,
    resize=True,
):

    logger.info("Get displacements using block mathching")
    args = (
        (im, reference_image, block_size, max_block_movement)
        for im in np.rollaxis(frames, 2)
    )
    num_frames = frames.shape[-1]

    y_size, x_size = reference_image.shape
    shape = (y_size // block_size, x_size // block_size)
    flows = np.zeros((shape[0], shape[1], 2, num_frames))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, uv in tqdm.tqdm(
            enumerate(executor.map(flow_map, args)),
            desc="Compute displacement",
            total=num_frames,
        ):
            flows[:, :, :, i] = uv

    if resize:

        new_shape: Tuple[int, int] = reference_image.shape[:2]
        int_flows = np.zeros((new_shape[0], new_shape[1], 2, num_frames))
        int_flows[:, :, 0, :] = resize_frames(flows[:, :, 0, :], new_shape=new_shape)
        int_flows[:, :, 1, :] = resize_frames(flows[:, :, 1, :], new_shape=new_shape)
        flows = int_flows

    return flows
