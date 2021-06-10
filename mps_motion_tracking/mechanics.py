import logging
from typing import Union

try:
    from functools import cached_property  # type: ignore
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
    dudx = da.gradient(displacement[:, :, :, 0], axis=0)
    dudy = da.gradient(displacement[:, :, :, 0], axis=1)
    dvdx = da.gradient(displacement[:, :, :, 1], axis=0)
    dvdy = da.gradient(displacement[:, :, :, 1], axis=1)

    return (1 / dx) * da.moveaxis(
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


    """

    def __init__(
        self,
        u: fs.VectorFrameSequence,
    ):
        assert isinstance(u, fs.VectorFrameSequence)
        self.u = u

    @property
    def dx(self) -> float:
        return self.u.dx

    @property
    def scale(self) -> float:
        return self.u.scale

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
        return (
            f"{self.__class__.__name__}(u={self.u}, dx={self.dx}, scale={self.scale})"
        )

    @property
    def du(self) -> fs.TensorFrameSequence:
        return fs.TensorFrameSequence(
            compute_gradients(self.u._array, dx=self.dx), dx=1.0, scale=self.scale
        )

    @property
    def F(self) -> fs.TensorFrameSequence:
        return fs.TensorFrameSequence(
            self.du.array + da.eye(2)[None, None, None, :, :], dx=1.0, scale=self.scale
        )

    @property
    def E(self) -> fs.TensorFrameSequence:
        return fs.TensorFrameSequence(
            compute_green_lagrange_strain_tensor(self.F.array), dx=1.0, scale=self.scale
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
