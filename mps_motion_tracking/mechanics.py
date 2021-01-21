import logging
import time
from typing import Union

import dask.array as da
import numpy as np

Array = Union[da.Array, np.ndarray]
logger = logging.getLogger(__name__)


def compute_gradients(displacement: Array, dx=1):
    """
    Computes gradients u_x, u_y, v_x, v_y from values in data

    Args:
        displacement - numpy array of dimensions T x X x Y x 2
        dx - float; spatial difference between two points/blocks

    Returns:
        numpy array of dimensions T x X x Y x 2 x 2

    """
    logger.info("Compute gradient")
    dudx = 1 / dx * da.gradient(displacement[:, :, :, 0], axis=1)
    dudy = 1 / dx * da.gradient(displacement[:, :, :, 0], axis=2)
    dvdx = 1 / dx * da.gradient(displacement[:, :, :, 1], axis=1)
    dvdy = 1 / dx * da.gradient(displacement[:, :, :, 1], axis=2)

    return da.moveaxis(
        da.moveaxis(da.stack([[dudx, dudy], [dvdx, dvdy]]), 0, -1), 0, -1
    )


def compute_green_lagrange_strain_tensor(F: Array):
    """
    Computes the transpose along the third dimension of data; if data
    represents the deformation gradient tensor F (over time and 2 spatial
    dimensions) this corresponds to the Green-Lagrange strain tensor E.

    Args:
        numpy array of dimensions T x X x Y x 2 x 2

    Returns
        numpy array of dimensions T x X x Y x 2 x 2

    """
    logger.info("Compute Green Lagrange strain")
    F_t = da.transpose(F, (0, 1, 2, 4, 3))
    C = da.matmul(F, F_t)
    E = 0.5 * (C - da.eye(2)[None, None, None, :, :])

    return E


def princial_values(A: Array) -> np.ndarray:
    """
    Return the larges eigenvalue of a symmetric matrix.
    I A is a strain tensor this is the same as the principal strain

    Parameters
    ----------
    A : array
        A numpy array or dask array consiting of the tensor

    Returns
    -------
    np.ndarray
        Eigenvalues
    """
    logger.info("Compute principal values")

    if isinstance(A, da.Array):
        t = time.perf_counter()
        logger.info("Compute dask array")
        A = A.compute()
        logger.info(f"Computed dask array in {time.perf_counter() -t:.3f} seconds")

    # E is symmetric so we can use this function which returns eigenvalues sored in acsending order
    t = time.perf_counter()
    eigenvalues = np.linalg.eigvalsh(A)
    logger.info(f"Princical values computed in {time.perf_counter() - t:.3f} seconds")

    return eigenvalues


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

    def __init__(self, u: Array):
        logger.info("Convert displacement to dask array")
        self.u = da.from_array(np.rollaxis(u, -1))

    @property
    def u_norm(self):
        """Norm of displacement at all pixels and all time points"""
        return da.linalg.norm(self.u, axis=-1)

    @property
    def u_mean_norm(self):
        """Mean of norm of displacement at all pixels and all time points"""
        return da.linalg.norm(self.u, axis=-1).mean((1, 2))

    def u_mean(self, axis=None):
        """Mean of displacement at all pixels and all time points at axis"""
        return self.u[..., axis].mean((1, 2))

    def u_max(self, axis=None):
        """Max displacement over all time points at all pixels and all time points at axis"""
        return self.u[..., axis].max(0)

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
            f"width={self.width})"
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(u = {self.u})"

    def du(self, recompute=False) -> da.Array:
        return compute_gradients(self.u)

    def F(self, recompute=False) -> da.Array:
        return self.du() + np.eye(2)[None, None, None, :, :]

    def E(self, recompute=False) -> da.Array:
        return compute_green_lagrange_strain_tensor(self.F())

    def principal_strain(self, k=0, recompute=False) -> np.ndarray:
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
        assert k in (0, 1)
        if not hasattr(self, "_principal_strain") or recompute:
            E = self.E()
            self._principal_strain = princial_values(E)
        return self._principal_strain[..., -1 + k]
