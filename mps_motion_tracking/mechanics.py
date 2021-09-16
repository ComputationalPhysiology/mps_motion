import logging
from typing import Optional
from typing import Union

try:
    from functools import cached_property  # type: ignore
except ImportError:
    # This is only supported in python 3.8 and above
    try:
        from cached_property import cached_property  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Please install cached_property - pip install cached_property",
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
        da.moveaxis(da.stack([[dudx, dudy], [dvdx, dvdy]]), 0, -1),
        0,
        -1,
    )


def compute_green_lagrange_strain_tensor(F: Array):
    logger.debug("Compute Green Lagrange strain")
    F_t = da.transpose(F, (0, 1, 2, 4, 3))
    C = da.matmul(F, F_t)
    E = 0.5 * (C - da.eye(2)[None, None, None, :, :])

    return E


def compute_velocity(u: Array, t: Array):

    time_axis = u.shape.index(len(t))
    assert time_axis == 2, "Time axis should be the first axis"
    assert u.shape[-1] == 2, "Final axis should be ux and uy"
    du = da.diff(u, axis=time_axis)
    dt = da.diff(t)

    # Need to have time axis
    return da.moveaxis(da.moveaxis(du, 2, 3) / dt, 2, 3).compute()


class Mechancis:
    def __init__(
        self,
        u: fs.VectorFrameSequence,
        t: Optional[Array] = None,
    ):
        """Craete a mechanics object

        Parameters
        ----------
        u : fs.VectorFrameSequence
            Displacment of shape height x width x time x 2
        t : Optional[Array], optional
            Time stamps of length (time), by default None.
            If not provided `t` will be an evenly spaced
            array with a step of 1.0. Note that `t` is only
            relevant when computing time derivaties such as velocity.
        """
        assert isinstance(u, fs.VectorFrameSequence)
        self.u = u
        self.t = t

    @property
    def t(self) -> Array:
        return self._t

    @t.setter
    def t(self, t: Optional[Array]) -> None:
        if t is None:
            t = np.arange(self.num_time_points)
        if not len(t) == self.num_time_points:
            raise RuntimeError(
                "Expected time stamps to have the same number of points at 'u'",
            )
        self._t = t

    @property
    def dx(self) -> float:
        return self.u.dx

    @property
    def scale(self) -> float:
        return self.u.scale

    @property
    def height(self) -> int:
        return self.u.shape[0]

    @property
    def width(self) -> int:
        return self.u.shape[1]

    @property
    def num_time_points(self) -> int:
        return self.u.shape[2]

    def __str__(self):
        return (
            f"{self.__class__.__name__} object with "
            f"num_time_points={self.num_time_points}, "
            f"height={self.height}, "
            f"width={self.width}, "
            f"dx={self.dx}"
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(u={self.u}, dx={self.dx}, scale={self.scale})"
        )

    @property
    def du(self) -> fs.TensorFrameSequence:
        return fs.TensorFrameSequence(
            compute_gradients(self.u.array, dx=self.dx),
            dx=1.0,
            scale=self.scale,
        )

    @property
    def F(self) -> fs.TensorFrameSequence:
        return fs.TensorFrameSequence(
            self.du.array + da.eye(2)[None, None, None, :, :],
            dx=1.0,
            scale=self.scale,
        )

    @property
    def E(self) -> fs.TensorFrameSequence:
        return fs.TensorFrameSequence(
            compute_green_lagrange_strain_tensor(self.F.array),
            dx=1.0,
            scale=self.scale,
        )

    @cached_property
    def velocity(self):
        return fs.VectorFrameSequence(compute_velocity(self.u.array, self.t))

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
