import logging
from pathlib import Path
from typing import List
from typing import Optional
from typing import Protocol

import dask.array as da
import numpy as np

from .utils import Array
from .utils import filter_vectors_par
from .utils import median_filter
from .utils import PathLike

try:
    import h5py

    has_h5py = True
except ImportError:
    has_h5py = False

logger = logging.getLogger(__name__)


class _Linalg(Protocol):
    @staticmethod
    def norm(array: Array, axis: int = 0) -> Array:
        ...


class NameSpace(Protocol):
    @property
    def linalg(self) -> _Linalg:
        ...

    @staticmethod
    def stack(arrs: List[Array], axis: int) -> Array:
        ...


class InvalidThresholdError(ValueError):
    pass


def check_threshold(
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    if (vmax and vmin) and vmax < vmin:
        raise InvalidThresholdError(f"Cannot have vmax < vmin, got {vmax=} and {vmin=}")


def threshold(
    array: Array,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    copy: bool = True,
) -> Array:
    """Threshold an array

    Parameters
    ----------
    array : Array
        The array
    vmin : Optional[float], optional
        Lower threshold value, by default None
    vmax : Optional[float], optional
        Upper threshold value, by default None
    copy : bool, optional
        Operative on the given array or use a copy, by default True

    Returns
    -------
    Array
        Inpute array with the lowest value beeing vmin and
        highest value begin vmax
    """
    assert len(array.shape) == 3
    check_threshold(vmin, vmax)
    if copy:
        array = array.copy()

    if vmax is not None:
        array[array > vmax] = vmax
    if vmin is not None:
        array[array < vmin] = vmin
    return array


def _handle_threshold_norm(norm_inds, ns, factor, norm_array, array):
    if norm_inds.any():
        if ns == da:
            norm_inds = norm_inds.compute()
        inds = np.stack([norm_inds, norm_inds], -1).flatten()
        values = (
            factor
            / ns.stack([norm_array[norm_inds], norm_array[norm_inds]], -1).flatten()
        )
        array[inds] *= values


def threshold_norm(
    array: Array,
    ns: NameSpace,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    copy: bool = True,
) -> Array:
    """Threshold an array of vectors based on the
    norm of the vectors.

    For example if the vectors are displacement then
    you can use this function to scale all vectors so
    that the magnitudes are within `vmin` and `vmax`

    Parameters
    ----------
    array : Array
        The input array which is 4D and the last dimension is 2.
    ns : NameSpace
        Wheter to use numpy or dask
    vmin : Optional[float], optional
        Lower bound on the norm, by default None
    vmax : Optional[float], optional
        Upper bound on the norm, by default None
    copy : bool, optional
        Wheter to operate on the input are or use a copy, by default True

    Returns
    -------
    Array
        The thresholded array
    """
    assert len(array.shape) == 4
    assert array.shape[3] == 2
    assert ns in [da, np]
    check_threshold(vmin, vmax)
    if copy:
        array = array.copy()
    shape = array.shape
    norm_array = ns.linalg.norm(array, axis=3).flatten()
    array = array.flatten()

    if vmax is not None:
        norm_inds = norm_array > vmax
        _handle_threshold_norm(norm_inds, ns, vmax, norm_array, array)
    if vmin is not None:
        norm_inds = norm_array < vmin
        _handle_threshold_norm(norm_inds, ns, vmin, norm_array, array)
    return array.reshape(shape)


class FrameSequence:
    """Object for holding a sequence of frames
    For example a component of a Tensor / Vector
    FrameSeqnecu

    """

    def __init__(self, array: Array, dx: float = 1.0, scale: float = 1.0):
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

        """
        assert isinstance(array, (da.core.Array, np.ndarray))
        self._ns = np if isinstance(array, np.ndarray) else da
        self._array = array
        self.dx = dx
        self.scale = scale
        self._h5file = None

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

    def __del__(self):
        if self._h5file is not None:
            self._h5file.close()

    def filter(self, size: int = 3) -> "FrameSequence":
        """Apply a median filter"""

        return FrameSequence(
            array=median_filter(self._array, size),
            dx=self.dx,
            scale=self.scale,
        )

    def save(self, path: PathLike) -> None:
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
            np.save(path, {"array": self.array_np, "scale": self.scale, "dx": self.dx})

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
        else:
            data.update(np.load(path, allow_pickle=True).item())

            if use_dask:
                data["array"] = da.from_array(data["array"])

        if "array" not in data:
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

    def threshold(self, vmin: Optional[float] = None, vmax: Optional[float] = None):
        array = threshold(self.array, vmin, vmax)
        return self.__class__(array, self.dx, self.scale)

    @property
    def array(self) -> Array:
        return self._array

    @property
    def array_np(self) -> np.ndarray:
        array = self.array
        if isinstance(self._array, da.core.Array):
            array = self.array.compute()  # type: ignore
        return array

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

    def mean(self) -> Array:
        return self.array.mean((0, 1)) * self.scale

    def max(self) -> Array:
        return self.array.max(2)

    def min(self) -> Array:
        return self.array.min(2)

    def compute(self) -> Array:
        return self.array_np

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.array.shape}, dx={self.dx}, scale={self.scale})"


class VectorFrameSequence(FrameSequence):
    """Object for holding a sequence of vectors.
    For example displacement

    """

    def __init__(self, array: Array, dx: float = 1.0, scale: float = 1.0):
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

    def filter(self, size: int = 5) -> "VectorFrameSequence":
        """Apply a median filter"""

        array = filter_vectors_par(self._array, size=size)

        return VectorFrameSequence(array=array, dx=self.dx, scale=self.scale)

    def threshold_norm(
        self,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> "VectorFrameSequence":
        array = threshold_norm(self.array, self._ns, vmin, vmax)
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
        return FrameSequence(self._array[:, :, :, 1], dx=self.dx, scale=self.scale)

    @property
    def y(self) -> FrameSequence:
        return FrameSequence(self._array[:, :, :, 0], dx=self.dx, scale=self.scale)


class TensorFrameSequence(FrameSequence):
    """Object for holding a sequence of tensors
    For example Green-Lagrange strain tensor
    and Cauchy stress tensor

    """

    def __init__(self, array: Array, dx: float = 1.0, scale: float = 1.0):
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
