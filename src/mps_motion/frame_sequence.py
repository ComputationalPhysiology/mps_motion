import logging
import weakref
from pathlib import Path
from typing import Optional

import dask.array as da
import numpy as np

from . import filters
from . import utils

try:
    import h5py

    has_h5py = True
except ImportError:
    has_h5py = False

logger = logging.getLogger(__name__)


@utils.jit(nopython=True)
def sum_along_time(arr, t, i, j):
    x = 0
    for k in range(t):
        x += (1 / t) * np.abs(arr[i, j, k, :, :, :] - arr[i, j, k, 1, 1, 1]).sum()
    return x


@utils.jit(nopython=True, parallel=True)
def reduce_sliding_window(arr):
    n, m, t = arr.shape[:3]
    new_arr = np.zeros((n, m))

    for i in utils.prange(n):
        for j in range(m):
            new_arr[i, j] = sum_along_time(arr, t, i, j)

    return new_arr


def close_file(h5file):
    if h5file is not None:
        h5file.close()


class FrameSequence:
    """Object for holding a sequence of frames
    For example a component of a Tensor / Vector
    FrameSeqnecu

    """

    def __init__(
        self,
        array: utils.Array,
        dx: float = 1.0,
        scale: float = 1.0,
        fill_value: float = 0.0,
    ):
        """Constructor

        Parameters
        ----------
        array : np.ndarray or dask array
            Frame sequence of shape
            (width, height, num_time_steps)
        dx : float
            The Physical size of one
            pixel in the Frame, by default 1.0. Note this can
            also incorporate translation from pixel size to
            physical size.
        scale : float
            Another factor that should be reflexted and averaging
        fill_value: float
            Value to fill in for bad values masked arrays if used,
            by default 0.0.

        """
        assert isinstance(array, (da.core.Array, np.ndarray))
        self._ns = np if isinstance(array, np.ndarray) else da
        self._array = array
        self.dx = dx
        self.scale = scale
        self._h5file = None
        self._fill_value = fill_value
        self._finalizer = weakref.finalize(self, close_file, self._h5file)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        return self._finalizer()

    def __getitem__(self, *args, **kwargs):
        return self.array.__getitem__(*args, **kwargs)

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        if self.scale != other.scale:
            return False
        if self.dx != other.dx:
            return False
        return self._ns.isclose(self.array, other.array).all()

    def _check_other(self, other):
        if isinstance(other, FrameSequence):
            other = other.array

        if not isinstance(other, (da.core.Array, np.ndarray)):
            raise TypeError(
                f"Invald type {type(other)}, exepcted FrameSequence or array",
            )
        if self.shape != other.shape:
            raise ValueError(
                f"Incompatible shape, got {other.shape}, expected {self.shape}",
            )
        return other

    def __add__(self, other):
        other = self._check_other(other)
        return self.__class__(self.array + other, dx=self.dx, scale=self.scale)

    def __sub__(self, other):
        other = self._check_other(other)
        return self.__class__(self.array - other, dx=self.dx, scale=self.scale)

    def __mul__(self, other):
        if not np.isscalar(other):
            raise TypeError(f"Can only multiply with a scalar value, got {type(other)}")

        return self.__class__(other * self.array, dx=self.dx, scale=self.scale)

    def __rmul__(self, other):
        if not np.isscalar(other):
            raise TypeError(f"Can only multiply with a scalar value, got {type(other)}")

        return self.__class__(other * self.array, dx=self.dx, scale=self.scale)

    @property
    def fill_value(self) -> float:
        """Value used for as a replacement for bad values
        in masked arrays. This will only be used if you
        apply a mask to the array"""
        return self._fill_value

    @fill_value.setter
    def fill_value(self, value: float) -> None:
        self._fill_value = value

    def filter(
        self,
        filter_type: filters.Filters = filters.Filters.median,
        size: int = 3,
        sigma: float = 1.0,
    ) -> "FrameSequence":
        """Apply a median filter"""

        return FrameSequence(
            array=filters.apply_filter(
                self._array,
                size=size,
                sigma=sigma,
                filter_type=filter_type,
            ),
            dx=self.dx,
            scale=self.scale,
        )

    def save(self, path: utils.PathLike) -> None:
        path = Path(path)

        if path.is_file():
            path.unlink()

        suffixes = [".h5", ".npy"]
        msg = f"Expected suffix to be one of {suffixes}, got {path.suffix}"
        assert path.suffix in suffixes, msg
        if path.suffix == ".h5":
            if not has_h5py:
                raise IOError("Cannot save to HDF5 format. Please install h5py")

            with h5py.File(path, "w") as f:
                dataset = f.create_dataset("array", data=self.array_np)
                attr_manager = h5py.AttributeManager(dataset)
                attr_manager.create("scale", str(self.scale))
                attr_manager.create("dx", str(self.dx))
        else:
            np.save(
                path,
                {
                    "array": self.array_np,
                    "scale": self.scale,
                    "dx": self.dx,
                },  # type:ignore
            )

    @classmethod
    def from_file(cls, path, use_dask=True):

        path = Path(path)
        if not path.is_file():
            raise IOError(f"File {path} foes not exist")

        suffixes = [".h5", ".npy"]
        msg = f"Expected suffix to be one of {suffixes}, got {path.suffix}"
        assert path.suffix in suffixes, msg
        data = {}
        h5file = None
        if path.suffix == ".h5":
            h5file = h5py.File(path, "r")
            try:
                if "array" in h5file:
                    if use_dask:
                        data["array"] = da.from_array(h5file["array"])
                    else:
                        data["array"] = h5file["array"][...]
                    data.update(
                        dict(
                            zip(
                                h5file["array"].attrs.keys(),
                                map(float, h5file["array"].attrs.values()),
                            ),
                        ),
                    )
                    data["dx"] = float(h5file["array"].attrs.get("dx", 1))
                    data["scale"] = float(h5file["array"].attrs.get("scale", 1))
            except Exception:
                h5file.close()
        else:
            data.update(np.load(path, allow_pickle=True).item())

            if use_dask:
                data["array"] = da.from_array(data["array"])

        if "array" not in data:
            if h5file is not None:
                h5file.close()
            raise IOError(f"Unable to load data from file {path}")

        obj = cls(**data)
        obj._h5file = h5file
        return obj

    def local_averages(self, N: int, background_correction: bool = False):
        """Compute averages in local regions

        Parameters
        ----------
        N : int
            Number of regions along major axis
        background_correction : bool
            If true apply background correction algorithm
            to remove drift.

        Returns
        -------
        np.ndarray
            The local averages
        """
        try:
            from mps.analysis import local_averages
        except ImportError as ex:
            msg = "Please install the mps package for computing local averages"
            raise ImportError(msg) from ex

        return local_averages(
            self.array_np,
            np.arange(self.shape[2]),
            background_correction=background_correction,
            N=N,
        )

    def amplitude_mask(self, threshold: float = 30.0) -> np.ndarray:
        logger.info("Find mask for filtering based on amplitude")
        x = np.lib.stride_tricks.sliding_window_view(self.array_np, (3, 3, 3)).squeeze()
        y = reduce_sliding_window(x)
        mask = np.zeros(self.shape[:2], dtype=bool)
        mask[1:-1, 1:-1][y > threshold] = True
        return mask

    def apply_mask(self, mask: np.ndarray) -> None:
        logger.debug("Apply mask")
        self._array = da.ma.masked_array(
            self.array,
            np.tile(mask[..., np.newaxis], self.shape[2:]),
            fill_value=0.0,
        )

    def threshold(self, vmin: Optional[float] = None, vmax: Optional[float] = None):
        array = filters.threshold(self.array, vmin, vmax)
        return self.__class__(array, self.dx, self.scale)

    @property
    def array(self) -> utils.Array:
        return self._array

    @property
    def array_np(self) -> np.ndarray:
        array = self.array
        if isinstance(self._array, da.core.Array):
            array = self.array.compute()  # type: ignore
        return utils.unmask(array, fill_value=self.fill_value)

    @property
    def dx(self):
        return self._dx

    @dx.setter
    def dx(self, dx):
        assert dx > 0, "dx has to be positive"
        self._dx = dx

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale):
        assert scale > 0, "scale has to be positive"
        self._scale = scale

    @property
    def original_shape(self):
        w, h = self.shape[:2]
        return (int(w * self.dx), int(h * self.dx), *self.shape[2:])

    @property
    def shape(self):
        return self.array.shape

    def mean(self) -> utils.Array:
        return self.array.mean((0, 1)) * self.scale

    def max(self) -> utils.Array:
        return self.array.max(2) * self.scale

    def min(self) -> utils.Array:
        return self.array.min(2) * self.scale

    def compute(self) -> utils.Array:
        return self.array_np * self.scale

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.array.shape}, dx={self.dx}, scale={self.scale})"


class VectorFrameSequence(FrameSequence):
    """Object for holding a sequence of vectors.
    For example displacement

    """

    def __init__(self, array: utils.Array, dx: float = 1.0, scale: float = 1.0):
        """Constructor

        Parameters
        ----------
        array : np.ndarray or dask array
            Tensor sequence of shape
            (width, height, num_time_steps, 2)
        dx : float
            The Physical size of one
            pixel in the Frame, by default 1.0. Note this can
            also incorporate translation from pixel size to
            physical size.
        scale : float
            Another factor that should be reflexted and averaging

        """
        super().__init__(array, dx, scale)
        assert len(array.shape) == 4
        assert array.shape[3] == 2

    def filter(
        self,
        filter_type: filters.Filters = filters.Filters.median,
        size: int = 3,
        sigma: float = 1.0,
    ) -> "VectorFrameSequence":
        """Apply a filter"""

        array = filters.filter_vectors_par(
            self._array,
            size=size,
            sigma=sigma,
            filter_type=filter_type,
        )

        return VectorFrameSequence(array=array, dx=self.dx, scale=self.scale)

    def spline_smooth(self):
        arr = filters.spline_smooth(self.array)
        return VectorFrameSequence(da.from_array(arr), scale=self.scale, dx=self.dx)

    # def cartToPolar(self):

    #     mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=True)

    def apply_mask(self, mask: np.ndarray) -> None:
        self._array = da.ma.masked_array(
            self.array,
            np.tile(mask[..., np.newaxis, np.newaxis], self.shape[2:]),
            fill_value=0.0,
        )

    def threshold_norm(
        self,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> "VectorFrameSequence":
        array = filters.threshold_norm(self.array, self._ns, vmin, vmax)
        return VectorFrameSequence(array, self.dx, self.scale)

    def norm(self) -> FrameSequence:
        return FrameSequence(
            self._ns.linalg.norm(self._array, axis=3),
            dx=self.dx,
            scale=self.scale,
        )

    def angle(self) -> FrameSequence:
        return FrameSequence(
            self._ns.arctan2(self._array[:, :, :, 0], self._array[:, :, :, 1]),
            dx=self.dx,
            scale=self.scale,
        )

    @property
    def x(self) -> FrameSequence:
        return FrameSequence(self._array[:, :, :, 0], dx=self.dx, scale=self.scale)

    @property
    def y(self) -> FrameSequence:
        return FrameSequence(self._array[:, :, :, 1], dx=self.dx, scale=self.scale)


class TensorFrameSequence(FrameSequence):
    """Object for holding a sequence of tensors
    For example Green-Lagrange strain tensor
    and Cauchy stress tensor

    """

    def __init__(self, array: utils.Array, dx: float = 1.0, scale: float = 1.0):
        """Constructor

        Parameters
        ----------
        array : np.ndarray or dask array
            Tensor sequence of shape
            (width, height, num_time_steps, 2, 2)
        dx : float
            The Physical size of one
            pixel in the Frame, by default 1.0. Note this can
            also incorporate translation from pixel size to
            physical size.
        scale : float
            Another factor that should be reflexted and averaging

        """
        super().__init__(array, dx, scale)
        self._ns = np if isinstance(array, np.ndarray) else da
        assert len(array.shape) == 5
        assert array.shape[3] == array.shape[4] == 2

    def norm(self) -> FrameSequence:
        return FrameSequence(
            self._ns.linalg.norm(self._array, axis=(3, 4)),
            dx=self.dx,
            scale=self.scale,
        )

    @property
    def x(self) -> FrameSequence:
        return FrameSequence(self._array[:, :, :, 1, 1], dx=self.dx, scale=self.scale)

    @property
    def y(self) -> FrameSequence:
        return FrameSequence(self._array[:, :, :, 0, 0], dx=self.dx, scale=self.scale)

    @property
    def xy(self) -> FrameSequence:
        return FrameSequence(self._array[:, :, :, 1, 0], dx=self.dx, scale=self.scale)

    @property
    def yx(self) -> FrameSequence:
        return FrameSequence(self.array[:, :, :, 0, 1], dx=self.dx, scale=self.scale)

    def compute_eigenvalues(self) -> VectorFrameSequence:
        return VectorFrameSequence(
            np.linalg.eigvalsh(self.array_np),
            dx=self.dx,
            scale=self.scale,
        )
