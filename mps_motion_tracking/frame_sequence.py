from pathlib import Path
from typing import Union

import dask.array as da
import numpy as np

try:
    import h5py

    has_h5py = True
except ImportError:
    has_h5py = False

Array = Union[da.core.Array, np.ndarray]
PathStr = Union[Path, str]


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
        self._array = array
        self.dx = dx
        self.scale = scale

    def __getitem__(self, *args, **kwargs):
        return self.array.__getitem__(*args, **kwargs)

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False

        if self.scale != other.scale:
            return False
        if self.dx != other.dx:
            return False
        if (self.array == other.array).all():
            return True
        return False

    def save(self, path: PathStr) -> None:
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
        if path.suffix == ".h5":
            with h5py.File(path) as f:
                if "array" in f:
                    data["array"] = f["array"][...]
                    data.update(
                        dict(
                            zip(
                                f["array"].attrs.keys(),
                                map(float, f["array"].attrs.values()),
                            )
                        )
                    )
        else:
            data.update(np.load(path, allow_pickle=True).item())

        if "array" not in data:
            raise IOError(f"Unable to load data from file {path}")

        if use_dask:
            array = da.from_array(data["array"])
            data["array"] = array

        return cls(**data)

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
        self._ns = np if isinstance(array, np.ndarray) else da
        assert len(array.shape) == 4
        assert array.shape[3] == 2

    def norm(self) -> FrameSequence:
        return FrameSequence(
            self._ns.linalg.norm(self._array, axis=3), dx=self.dx, scale=self.scale
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
            self._ns.linalg.norm(self._array, axis=(3, 4)), dx=self.dx, scale=self.scale
        )

    @property
    def x(self) -> FrameSequence:
        return FrameSequence(self._array[:, :, :, 0, 0], dx=self.dx, scale=self.scale)

    @property
    def y(self) -> FrameSequence:
        return FrameSequence(self._array[:, :, :, 1, 1], dx=self.dx, scale=self.scale)

    @property
    def xy(self) -> FrameSequence:
        return FrameSequence(self._array[:, :, :, 1, 0], dx=self.dx, scale=self.scale)

    @property
    def yx(self) -> FrameSequence:
        return FrameSequence(self.array[:, :, :, 0, 1], dx=self.dx, scale=self.scale)

    def compute_eigenvalues(self) -> VectorFrameSequence:
        return VectorFrameSequence(
            np.linalg.eigvalsh(self.array_np), dx=self.dx, scale=self.scale
        )
