import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from . import block_matching, dualtvl10, farneback, lucas_kanade, utils

logger = logging.getLogger(__name__)

DENSE_FLOW_ALGORITHMS = ["farneback", "dualtvl10"]
SPARSE_FLOW_ALGORITHMS = ["lucas_kanade", "block_matching"]


def _check_algorithm(alg, algs):
    msg = f"Expected flow algorithm to be one of {algs}, got {alg}"
    if alg not in algs:
        raise ValueError(msg)


def get_mps_displacements(path, flow_algorithm, **options):
    import mps

    data = mps.MPS(path)
    if flow_algorithm in DENSE_FLOW_ALGORITHMS:
        motion = DenseOpticalFlow(data, flow_algorithm=flow_algorithm, **options)
    elif flow_algorithm in SPARSE_FLOW_ALGORITHMS:
        motion = DenseOpticalFlow(data, flow_algorithm=flow_algorithm, **options)
    return motion.get_displacements()


class OpticalFlow(ABC):
    def __init__(
        self,
        data: utils.MPSData,
        flow_algorithm: str = "",
        reference_frame: Union[int, str] = 0,
        **options,
    ):
        self.data = data

        self.flow_algorithm = flow_algorithm
        self.options = options
        self.set_reference_frame(reference_frame)
        self._handle_algorithm()

    @abstractmethod
    def _handle_algorithm(self):
        pass

    @abstractmethod
    def get_displacements(self):
        pass

    @abstractmethod
    def get_velocities(self):
        pass

    @abstractmethod
    def dump(self, filname):
        pass

    @property
    def reference_frame(self) -> str:
        return self._reference_frame

    @property
    def reference_image(self) -> np.ndarray:
        return self._reference_image

    def set_reference_frame(self, reference_frame):

        try:
            reference_frame = int(reference_frame)
            self._reference_frame = str(reference_frame)
            self._reference_image = self.data.frames[:, :, int(reference_frame)]

        except ValueError:
            refs = ["min", "max", "median", "mean"]
            msg = (
                "Expected reference frame to be an integer or one of "
                f"{refs}, got {reference_frame}"
            )
            if reference_frame not in refs:
                raise ValueError(msg)
            self._reference_frame = reference_frame
            self._reference_image = getattr(np, reference_frame)(
                self.data.frames, axis=2
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"data={self.data}, flow_algorithm={self.flow_algorithm})"
        )


class DenseOpticalFlow(OpticalFlow):
    def __init__(
        self,
        data: utils.MPSData,
        flow_algorithm: str = "farneback",
        reference_frame: Union[int, str] = 0,
        **options,
    ):
        super().__init__(data, flow_algorithm, reference_frame, **options)

    def _handle_algorithm(self):
        _check_algorithm(self.flow_algorithm, DENSE_FLOW_ALGORITHMS)

        if self.flow_algorithm == "farneback":
            self._flow = farneback.flow
            self._flow_map = farneback.flow_map
            self._get_displacements = farneback.get_displacements
            self._get_velocities = farneback.get_velocities
            options = farneback.default_options()

        elif self.flow_algorithm == "dualtvl10":
            self._flow = dualtvl10.flow
            self._flow_map = dualtvl10.flow_map
            self._get_displacements = dualtvl10.get_displacements
            self._get_velocities = dualtvl10.get_velocities
            options = dualtvl10.default_options()

        self.options.update(options)

    def get_displacements(self, recompute=False):
        if not hasattr(self, "_displacement") or recompute:
            self._disp = self._get_displacements(
                self.data.frames, self.reference_image, **self.options
            )
        return self._disp

    def get_velocities(self):
        raise NotImplementedError

    def dump(self):
        raise NotImplementedError


class SparseOpticalFlow(OpticalFlow):
    def __init__(
        self,
        data: utils.MPSData,
        flow_algorithm: str = "lucas_kanade",
        reference_frame: Union[int, str] = 0,
        **options,
    ):
        super().__init__(data, flow_algorithm, reference_frame, **options)

    def _handle_algorithm(self):
        _check_algorithm(self.flow_algorithm, SPARSE_FLOW_ALGORITHMS)

        if self.flow_algorithm == "lucas_kanade":
            self._flow = lucas_kanade.flow
            self._flow_map = lucas_kanade.flow_map
            self._get_displacements = lucas_kanade.get_displacements
            self._get_velocities = None  # lucas_kanade.get_velocities
            options = lucas_kanade.default_options()
        elif self.flow_algorithm == "block_matching":
            self._flow = block_matching.flow
            self._flow_map = block_matching.flow_map
            self._get_displacements = block_matching.get_displacements
            self._get_velocities = None  # block_matching.get_velocities
            options = block_matching.default_options()

        self.options.update(options)

    def get_displacements(self, recompute=False):
        if not hasattr(self, "_displacement") or recompute:
            self._disp = self._get_displacements(
                self.data.frames, self.reference_image, **self.options
            )
        return self._disp

    def get_velocities(self):
        raise NotImplementedError

    def dump(self):
        raise NotImplementedError
