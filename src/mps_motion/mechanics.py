import functools
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

from dask.diagnostics import ProgressBar
import dask.array as da
import numpy as np
import dask
from scipy import interpolate

from . import frame_sequence as fs

Array = Union[da.Array, np.ndarray]
logger = logging.getLogger(__name__)


def compute_gradients(displacement: Array, dx=1) -> da.Array:
    """Compute gradients of the displacement

    Parameters
    ----------
    displacement : Array
        Displacement vectors
    dx : int, optional
        Number of difference steps for gradient computations, by default 1

    Returns
    -------
    da.Array
        Gradients
    """
    logger.info("Compute gradient using spline interpolation")
    shape = displacement.shape
    x = dx * np.arange(shape[0])
    y = dx * np.arange(shape[1])

    Uxs = []
    Uys = []
    for t in range(shape[2]):
        ux = displacement[:, :, t, 0]
        Uxs.append(dask.delayed(interpolate.RectBivariateSpline)(x, y, ux))
        uy = displacement[:, :, t, 1]
        Uys.append(dask.delayed(interpolate.RectBivariateSpline)(x, y, uy))

    dudxs = []
    dudys = []
    dvdxs = []
    dvdys = []

    logger.info("Compute interpolant Ux")
    with ProgressBar():
        Uxs = dask.compute(*Uxs)

    logger.info("Compute interpolant Uy")
    with ProgressBar():
        Uys = dask.compute(*Uys)

    for (
        Ux,
        Uy,
    ) in zip(Uxs, Uys):
        dudxs.append(dask.delayed(Ux)(x, y, dx=1).T)
        dudys.append(dask.delayed(Ux)(x, y, dy=1).T)
        dvdxs.append(dask.delayed(Uy)(x, y, dx=1).T)
        dvdys.append(dask.delayed(Uy)(x, y, dy=1).T)

    logger.info("Compute dudx")
    with ProgressBar():
        Dudx = da.stack(*dask.compute(dudxs))
    logger.info("Compute dudy")
    with ProgressBar():
        Dudy = da.stack(*dask.compute(dudys))
    logger.info("Compute dvdx")
    with ProgressBar():
        Dvdx = da.stack(*dask.compute(dvdxs))
    logger.info("Compute dvdy")
    with ProgressBar():
        Dvdy = da.stack(*dask.compute(dvdys))

    logger.info("Stack arrays")
    Du = da.stack([[Dudx, Dudy], [Dvdx, Dvdy]]).T
    logger.info("Done computing gradient")
    return Du


def compute_green_lagrange_strain_tensor(F: Array) -> da.Array:
    r"""Compute Green-Lagrange strain tensor

    .. math::

        \mathbf{E} = \frac{1}{2} \left( F^T F - I \right)

    Parameters
    ----------
    F : Array
        Deformation gradient

    Returns
    -------
    da.Array
        Green-Lagrange strain tensor
    """
    logger.debug("Compute Green Lagrange strain")
    F_t = da.transpose(F, (0, 1, 2, 4, 3))
    C = da.matmul(F_t, F)
    E = 0.5 * (C - da.eye(2)[None, None, None, :, :])

    return E


def compute_velocity(u: Array, t: Array, spacing: int = 1) -> da.Array:
    """Compute velocity from displacement

    Parameters
    ----------
    u : Array
        Displacement vectors
    t : Array
        Time stamps
    spacing : int, optional
        Number of steps between time steps to compute velocities, by default 1

    Returns
    -------
    da.Array
        Velocity
    """
    time_axis = u.shape.index(len(t))
    assert time_axis == 2, "Time axis should be the third axis"
    assert u.shape[-1] == 2, "Final axis should be ux and uy"
    assert spacing > 0, "Spacing must be a positive integer"

    if spacing == 1:
        dt = da.diff(t)
        du = da.diff(u, axis=time_axis)
    else:
        dt = t[spacing:] - t[:-spacing]
        du = u[:, :, spacing:, :] - u[:, :, :-spacing, :]

    # # Need to have time axis
    return da.moveaxis(da.moveaxis(du, 2, 3) / dt, 2, 3)


def compute_displacement(v: Array, t: Array, ref_index=0, spacing: int = 1) -> da.Array:
    """Compute displacement from velocity

    Parameters
    ----------
    v : Array
        Velocities
    t : Array
        time stamps
    ref_index : int, optional
        Index to be used as reference frame, by default 0
    spacing : int, optional
        Spacing used to compute velocities, by default 1

    Returns
    -------
    da.Array
        Displacement

    Raises
    ------
    NotImplementedError
        If spacing is different from 1
    """
    if spacing != 1:
        raise NotImplementedError("Only implemented for the case when spacing is 1")
    zero = da.zeros((v.shape[0], v.shape[1], 1, v.shape[3]))
    dt = np.diff(t)
    vdt = da.apply_along_axis(lambda x: x * dt, axis=2, arr=v)

    vdt_low = vdt[:, :, :ref_index, :]
    vdt_high = vdt[:, :, ref_index:, :]
    U = da.concatenate(
        (
            da.flip(da.cumsum(-da.flip(vdt_low), axis=2)),
            zero,
            da.cumsum(vdt_high, axis=2),
        ),
        axis=2,
    )

    return U


class Mechanics:
    def __init__(
        self,
        u: fs.VectorFrameSequence,
        t: Optional[Array] = None,
    ):
        """Create a mechanics object

        Parameters
        ----------
        u : fs.VectorFrameSequence
            Displacement of shape height x width x time x 2
        t : Optional[Array], optional
            Time stamps of length (time), by default None.
            If not provided `t` will be an evenly spaced
            array with a step of 1.0. Note that `t` is only
            relevant when computing time derivatives such as velocity.
        """
        assert isinstance(u, fs.VectorFrameSequence)
        self._u = u
        self.t = t

    @property
    def u(self) -> fs.VectorFrameSequence:
        """Displacement field"""
        return self._u

    @property
    def t(self) -> Array:
        """Time stamps"""
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

    @cached_property
    def du(self) -> fs.TensorFrameSequence:

        try:
            du = compute_gradients(self.u.array, dx=self.dx)
        except ValueError:
            # We probably need to rechunk
            if isinstance(self.u._array, da.Array):
                logger.warning("Problems with computing gradient - try rechunking")
                self.u._array = self.u.array.rechunk()  # type:ignore
                du = compute_gradients(self.u.array, dx=self.dx)
            else:
                raise
        return fs.TensorFrameSequence(
            du,
            dx=self.dx,
            scale=self.scale,
        )

    @property
    def F(self) -> fs.TensorFrameSequence:
        """Deformation gradient"""
        return fs.TensorFrameSequence(
            self.du.array + da.eye(2)[None, None, None, :, :],
            dx=self.dx,
            scale=self.scale,
        )

    @property
    def E(self) -> fs.TensorFrameSequence:
        """Green-Lagrange strain tensor"""
        return fs.TensorFrameSequence(
            compute_green_lagrange_strain_tensor(self.F.array),
            dx=self.dx,
            scale=self.scale,
        )

    @functools.lru_cache
    def velocity(self, spacing: int = 1) -> fs.VectorFrameSequence:
        """Velocity field"""
        return fs.VectorFrameSequence(
            compute_velocity(self.u.array, self.t, spacing=spacing),
            dx=self.dx,
            scale=self.scale,
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
