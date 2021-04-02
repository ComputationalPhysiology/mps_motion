from typing import Union

import dask.array as da
import numpy as np

Array = Union[da.core.Array, np.ndarray]


class FrameSequence:
    """Object for holding a sequence of frames
    For example a component of a Tensor / Vector
    FrameSeqnecu

    """

    def __init__(self, array: Array, dx: float = 1.0):
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

        """
        assert isinstance(array, (da.core.Array, np.ndarray))
        self._array = array
        self.dx = dx

    def __getitem__(self, *args, **kwargs):
        return self.array.__getitem__(*args, **kwargs)

    @property
    def array(self):
        return self._array * self.dx

    @property
    def dx(self):
        return self._dx

    @dx.setter
    def dx(self, dx):
        assert dx > 0, "dx has to be positive"
        self._dx = dx

    @property
    def original_shape(self):
        w, h = self.shape[:2]
        return (int(w * self.dx), int(h * self.dx), *self.shape[2:])

    @property
    def shape(self):
        return self.array.shape

    def mean(self) -> Array:
        return self.array.mean((0, 1)) / self.dx

    def max(self) -> Array:
        return self.array.max(2)

    def __eq__(self, other) -> bool:
        return bool(np.isclose(self.array, other.array).all())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.array.shape}, dx={self.dx})"


class VectorFrameSequence(FrameSequence):
    """Object for holding a sequence of vectors.
    For example displacement

    """

    def __init__(self, array: Array, dx: float = 1.0):
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

        """
        super().__init__(array, dx)
        self._ns = np if isinstance(array, np.ndarray) else da
        assert len(array.shape) == 4
        assert array.shape[3] == 2

    def norm(self) -> FrameSequence:
        return FrameSequence(self._ns.linalg.norm(self._array, axis=3), dx=self.dx)

    @property
    def x(self) -> FrameSequence:
        return FrameSequence(self._array[:, :, :, 0], dx=self.dx)

    @property
    def y(self) -> FrameSequence:
        return FrameSequence(self._array[:, :, :, 1], dx=self.dx)


class TensorFrameSequence(FrameSequence):
    """Object for holding a sequence of tensors
    For example Green-Lagrange strain tensor
    and Cauchy stress tensor

    """

    def __init__(self, array: Array, dx: float = 1.0):
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

        """
        super().__init__(array, dx)
        self._ns = np if isinstance(array, np.ndarray) else da
        assert len(array.shape) == 5
        assert array.shape[3] == array.shape[4] == 2

    def norm(self) -> FrameSequence:
        return FrameSequence(self._ns.linalg.norm(self._array, axis=(3, 4)), dx=self.dx)

    @property
    def x(self) -> FrameSequence:
        return FrameSequence(self._array[:, :, :, 0, 0], dx=self.dx)

    @property
    def y(self) -> FrameSequence:
        return FrameSequence(self._array[:, :, :, 1, 1], dx=self.dx)

    @property
    def xy(self) -> FrameSequence:
        return FrameSequence(self._array[:, :, :, 1, 0], dx=self.dx)

    @property
    def yx(self) -> FrameSequence:
        return FrameSequence(self.array[:, :, :, 0, 1], dx=self.dx)

    def compute_eigenvalues(self) -> VectorFrameSequence:
        try:
            # If we have a dask array
            array = self._array.compute()  # type: ignore
        except AttributeError:
            array = self._array

        return VectorFrameSequence(np.linalg.eigvalsh(array), dx=self.dx)
