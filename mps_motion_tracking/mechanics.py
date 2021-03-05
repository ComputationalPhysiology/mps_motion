import logging
from typing import Union

try:
    from functools import cached_property
except ImportError:
    # This is only supported in python 3.8 and above
    try:
        from cached_property import cached_property  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Please install cached_property - pip install cached_property"
        ) from e


import dask.array as da
import numpy as np

from . import frame_sequence as fs

Array = Union[da.Array, np.ndarray]
logger = logging.getLogger(__name__)


def compute_gradients(displacement: Array, dx=1):
    logger.debug("Compute gradient")
    dudx = 1 / dx * da.gradient(displacement[:, :, :, 0], axis=0)
    dudy = 1 / dx * da.gradient(displacement[:, :, :, 0], axis=1)
    dvdx = 1 / dx * da.gradient(displacement[:, :, :, 1], axis=0)
    dvdy = 1 / dx * da.gradient(displacement[:, :, :, 1], axis=1)

    return da.moveaxis(
        da.moveaxis(da.stack([[dudx, dudy], [dvdx, dvdy]]), 0, -1), 0, -1
    )


def compute_green_lagrange_strain_tensor(F: Array):
    logger.debug("Compute Green Lagrange strain")
    F_t = da.transpose(F, (0, 1, 2, 4, 3))
    C = da.matmul(F, F_t)
    E = 0.5 * (C - da.eye(2)[None, None, None, :, :])

    return E


class Mechancis:
    """Class to compute mechanical quantities

    Parameters
    ----------
    u : Array
        numpy or dask array representing the displacements.
        u should be in the correct units, e.g um.
        It is assumed that axis are as follows:
        X x Y x 2 x T
    dx : float
        Physical size of one pixel in the Frame, by default 1.0

    """

    def __init__(self, u: Array, dx: float = 1):
        logger.debug("Convert displacement to dask array")
        self.u = fs.VectorFrameSequence(da.from_array(np.swapaxes(u, 2, 3)))
        self.dx = dx

    @property
    def width(self) -> int:
        return self.u.shape[2]

    @property
    def height(self) -> int:
        return self.u.shape[1]

    @property
    def num_time_points(self) -> int:
        return self.u.shape[0]

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"num_time_points={self.num_time_points}, "
            f"height={self.height}, "
            f"width={self.width}, "
            f"dx={self.dx})"
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(u={self.u}, dx={self.dx})"

    @property
    def du(self) -> fs.TensorFrameSequence:
        return fs.TensorFrameSequence(compute_gradients(self.u.array, dx=self.dx))

    @property
    def F(self) -> fs.TensorFrameSequence:
        return fs.TensorFrameSequence(self.du.array + da.eye(2)[None, None, None, :, :])

    @property
    def E(self) -> fs.TensorFrameSequence:
        return fs.TensorFrameSequence(
            compute_green_lagrange_strain_tensor(self.F.array)
        )

    @cached_property
    def principal_strain(self) -> fs.VectorFrameSequence:
        """Return the principal strain

        Parameters
        ----------
        k : int, optional
            Which component of the princial strain either 0 or 1, by default 0.
            0 correspond to the largets eigen value
        recompute : bool, optional
            If already computed set this to True if you want to recompute, by default False

        Returns
        -------
        np.ndarray
            The kth principal strain
        """
        return self.E.compute_eigenvalues()
