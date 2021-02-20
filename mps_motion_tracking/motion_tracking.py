import logging
from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np

from . import block_matching, dualtvl10, farneback, lucas_kanade, utils

logger = logging.getLogger(__name__)


class FLOW_ALGORITHMS(str, Enum):
    farneback = "farneback"
    dualtvl10 = "dualtvl10"
    lucas_kanade = "lucas_kanade"
    block_matching = "block_matching"


def _check_algorithm(alg):
    msg = f"Expected flow algorithm to be one of {FLOW_ALGORITHMS}, got {alg}"
    if alg not in FLOW_ALGORITHMS._member_names_:
        raise ValueError(msg)


def get_referenece_image(
    reference_frame, frames, time_stamps: Optional[np.ndarray] = None
) -> Tuple[str, np.ndarray]:
    try:
        reference_time = float(reference_frame)
        reference_str = str(reference_frame)

    except ValueError:
        refs = ["min", "max", "median", "mean"]
        msg = (
            "Expected reference frame to be an integer or one of "
            f"{refs}, got {reference_frame}"
        )
        if reference_frame not in refs:
            raise ValueError(msg)
        reference_str = reference_frame
        reference_image = getattr(np, reference_frame)(frames, axis=2)
    else:
        if time_stamps is None:
            raise ValueError("Please provide time stamps")
        try:
            reference_frame = next(
                i for i, t in enumerate(time_stamps) if t >= reference_time
            )
        except StopIteration:
            reference_frame = len(time_stamps) - 1

        reference_frame = min(reference_frame, len(time_stamps) - 1)
        reference_image = frames[:, :, int(reference_frame)]
    return reference_str, reference_image


class OpticalFlow:
    def __init__(
        self,
        data: utils.MPSData,
        flow_algorithm: str = "farneback",
        reference_frame: Union[int, str] = 0,
        **options,
    ):
        self.data = data

        self.flow_algorithm = flow_algorithm
        self.options = options
        self._reference_frame, self._reference_image = get_referenece_image(
            reference_frame, data.frames, data.time_stamps
        )
        self._handle_algorithm()

    def _handle_algorithm(self):
        _check_algorithm(self.flow_algorithm)

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

        elif self.flow_algorithm == "farneback":
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

    def get_displacements(
        self, recompute: bool = False, unit: str = "pixels", scale: float = 1.0
    ) -> np.ndarray:
        """Compute motion of all images relative to reference frame

        Parameters
        ----------
        recompute : bool, optional
            If allready computed set this to true if you want to
            recomputed, by default False
        unit : str, optional
            Either 'pixels' or 'um', by default "pixels".
            If using 'um' them the MPSData.info has to contain the
            key 'um_per_pixel'.
        scale : float, optional
            If less than 1.0, downsample images before estimating motion, by default 1.0

        Returns
        -------
        np.ndarray
            The displacements
        """
        assert unit in ["pixels", "um"]
        data = self.data
        reference_image = self.reference_image
        if scale > 1.0:
            raise ValueError("Cannot have scale larger than 1.0")
        if scale < 1.0:
            data = utils.resize_data(data, scale)
            _, reference_image = get_referenece_image(
                self.reference_frame, data.frames, data.time_stamps
            )

        if not hasattr(self, "_displacement") or recompute:
            self._disp = self._get_displacements(
                data.frames, reference_image, **self.options
            )
            if unit == "um":
                self._disp *= data.info.get("um_per_pixel", 1.0)
            else:
                if scale < 1.0:
                    self._disp /= np.sqrt(scale)

        return self._disp

    def get_velocities(self):
        raise NotImplementedError

    def dump(self):
        raise NotImplementedError

    @property
    def reference_frame(self) -> str:
        return self._reference_frame

    @property
    def reference_image(self) -> np.ndarray:
        return self._reference_image

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"data={self.data}, flow_algorithm={self.flow_algorithm})"
        )
