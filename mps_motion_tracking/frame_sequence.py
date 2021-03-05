from typing import Union

import dask.array as da
import numpy as np

Array = Union[da.Array, np.ndarray]


class FrameSequence:
    """Object for holding a sequence of frames
    For example a component of a Tensor / Vector
    FrameSeqnecu

    """

    def __init__(self, array: Array):
        """Constructor

        Parameters
        ----------
        array : np.ndarray or dask array
            Frame sequence of shape
            (width, height, num_time_steps)

        """
        assert isinstance(array, (da.core.Array, np.ndarray))
        self.array = array

    def __getitem__(self, *args, **kwargs):
        return self.array.__getitem__(*args, **kwargs)

    @property
    def shape(self):
        return self.array.shape

    def mean(self) -> Array:
        return self.array.mean((0, 1))

    def max(self) -> Array:
        return self.array.max(2)

    def __eq__(self, other) -> bool:
        return np.isclose(self.array, other.array).all()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.array.shape})"


class VectorFrameSequence(FrameSequence):
    """Object for holding a sequence of vectors.
    For example displacement

    """

    def __init__(self, array: Array):
        """Constructor

        Parameters
        ----------
        array : np.ndarray or dask array
            Tensor sequence of shape
            (width, height, num_time_steps, 2)

        """
        super().__init__(array)
        self._ns = np if isinstance(array, np.ndarray) else da
        assert len(array.shape) == 4
        assert array.shape[3] == 2

    def norm(self) -> FrameSequence:
        return FrameSequence(self._ns.linalg.norm(self.array, axis=3))

    @property
    def x(self) -> FrameSequence:
        return FrameSequence(self.array[:, :, :, 0])

    @property
    def y(self) -> FrameSequence:
        return FrameSequence(self.array[:, :, :, 1])


class TensorFrameSequence(FrameSequence):
    """Object for holding a sequence of tensors
    For example Green-Lagrange strain tensor
    and Cauchy stress tensor

    """

    def __init__(self, array: Array):
        """Constructor

        Parameters
        ----------
        array : np.ndarray or dask array
            Tensor sequence of shape
            (width, height, num_time_steps, 2, 2)

        """
        super().__init__(array)
        self._ns = np if isinstance(array, np.ndarray) else da
        assert len(array.shape) == 5
        assert array.shape[3] == array.shape[4] == 2

    def norm(self) -> FrameSequence:
        return FrameSequence(self._ns.linalg.norm(self.array, axis=(3, 4)))

    @property
    def x(self) -> FrameSequence:
        return FrameSequence(self.array[:, :, :, 0, 0])

    @property
    def y(self) -> FrameSequence:
        return FrameSequence(self.array[:, :, :, 1, 1])

    @property
    def xy(self) -> FrameSequence:
        return FrameSequence(self.array[:, :, :, 1, 0])

    @property
    def yx(self) -> FrameSequence:
        return FrameSequence(self.array[:, :, :, 0, 1])

    def compute_eigenvalues(self) -> VectorFrameSequence:
        try:
            # If we have a dask array
            array = self.array.compute()
        except AttributeError:
            array = self.array

        return VectorFrameSequence(np.linalg.eigvalsh(array))
