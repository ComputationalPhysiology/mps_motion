import logging
from enum import Enum
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import dask.array as da
import numpy as np

from . import block_matching
from . import dualtvl1
from . import farneback
from . import frame_sequence as fs
from . import lucas_kanade
from . import scaling
from . import utils


logger = logging.getLogger(__name__)


class FLOW_ALGORITHMS(str, Enum):
    farneback = "farneback"
    dualtvl1 = "dualtvl1"
    lucas_kanade = "lucas_kanade"
    block_matching = "block_matching"


def list_optical_flow_algorithm():
    return FLOW_ALGORITHMS._member_names_


class RefFrames(str, Enum):
    min = "min"
    max = "max"
    median = "median"
    mean = "mean"


def _check_algorithm(alg):
    msg = f"Expected flow algorithm to be one of {FLOW_ALGORITHMS._member_names_}, got {alg}"
    if alg not in FLOW_ALGORITHMS._member_names_:
        raise ValueError(msg)


def get_referenece_image(
    reference_frame: Union[int, str, RefFrames],
    frames: np.ndarray,
    time_stamps: Optional[np.ndarray] = None,
    smooth_ref_transition: bool = True,
) -> Tuple[str, np.ndarray, int]:
    """Get reference frame

    Parameters
    ----------
    reference_frame : Union[int, str, RefFrames]
        Either an integer of string representing the frame
        that should be used as reference
    frames : np.ndarray
        The image frames
    time_stamps : Optional[np.ndarray], optional
        The time stamps, by default None
    smooth_ref_transition : bool, optional
        If true, compute the mean frame of the three closest frames, by default True

    Returns
    -------
    Tuple[str, np.ndarray, int]
        (reference_str, reference_image, reference_frame_index)

    """

    reference_frame_index = 0
    try:
        reference_time = float(reference_frame)
        reference_str = str(reference_frame)

    except ValueError:
        refs = RefFrames._member_names_
        msg = (
            "Expected reference frame to be an integer or one of "
            f"{refs}, got {reference_frame}"
        )
        if str(reference_frame) not in refs:
            raise ValueError(msg)
        reference_str = str(reference_frame)
        reference_image = getattr(np, str(reference_frame))(frames, axis=2)
    else:
        if time_stamps is None:
            raise ValueError("Please provide time stamps")
        try:
            reference_frame_index = next(
                i for i, t in enumerate(time_stamps) if t >= reference_time
            )
        except StopIteration:
            reference_frame_index = len(time_stamps) - 1

        reference_frame_index = int(min(reference_frame_index, len(time_stamps) - 1))
        # Pick neighbouring index
        if smooth_ref_transition:
            if reference_frame_index == 0:
                reference_image = frames[
                    :,
                    :,
                    reference_frame_index : reference_frame_index + 3,
                ].mean(-1)
            elif reference_frame_index == len(time_stamps) - 1:
                reference_image = frames[
                    :,
                    :,
                    reference_frame_index - 2 : reference_frame_index + 1,
                ].mean(-1)
            else:
                reference_image = frames[
                    :,
                    :,
                    reference_frame_index - 1 : reference_frame_index + 2,
                ].mean(-1)
        else:
            reference_image = frames[:, :, reference_frame_index]

    return reference_str, reference_image, reference_frame_index


class OpticalFlow:
    def __init__(
        self,
        data: utils.MPSData,
        flow_algorithm: FLOW_ALGORITHMS = FLOW_ALGORITHMS.farneback,
        reference_frame: Union[int, str, RefFrames] = 0,
        filter_options: Optional[Dict[str, Any]] = None,
        data_scale: float = 1.0,
        smooth_ref_transition: bool = True,
        **options,
    ):
        self.data = data

        self.flow_algorithm = flow_algorithm

        (
            self._reference_frame,
            self._reference_image,
            self._reference_frame_index,
        ) = get_referenece_image(
            reference_frame,
            data.frames,
            data.time_stamps,
            smooth_ref_transition=smooth_ref_transition,
        )

        self._handle_algorithm(options)
        options["filter_options"] = filter_options or {}
        self._data_scale = data_scale

    @property
    def data_scale(self) -> float:
        return self._data_scale

    def _handle_algorithm(self, options):
        _check_algorithm(self.flow_algorithm)

        self._get_velocities = None
        if self.flow_algorithm == FLOW_ALGORITHMS.lucas_kanade:
            self._get_displacements = lucas_kanade.get_displacements
            self.options = lucas_kanade.default_options()

        elif self.flow_algorithm == FLOW_ALGORITHMS.block_matching:
            self._get_displacements = block_matching.get_displacements
            self.options = block_matching.default_options()

        elif self.flow_algorithm == FLOW_ALGORITHMS.farneback:
            self._get_displacements = farneback.get_displacements
            self.options = farneback.default_options()
            self._get_velocities = farneback.get_velocities

        elif self.flow_algorithm == FLOW_ALGORITHMS.dualtvl1:
            self._get_displacements = dualtvl1.get_displacements
            self.options = dualtvl1.default_options()

        self.options.update(options)

    def get_displacements(
        self,
        recompute: bool = False,
        unit: str = "um",
        scale: float = 1.0,
    ) -> fs.VectorFrameSequence:
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

        scaled_data = data
        if scale < 1.0:
            scaled_data = scaling.resize_data(data, scale)
            _, reference_image, _ = get_referenece_image(
                self.reference_frame,
                scaled_data.frames,
                scaled_data.time_stamps,
            )

        if not hasattr(self, "_displacement") or recompute:

            u = self._get_displacements(
                scaled_data.frames, reference_image, **self.options
            )
            dx = 1

            scale *= self.data_scale

            u /= scale

            if unit == "um":
                u *= scaled_data.info.get("um_per_pixel", 1.0)
                dx *= scaled_data.info.get("um_per_pixel", 1.0)
            else:
                u /= scale

            if not isinstance(u, da.Array):
                u = da.from_array(u)

            self._displacement = fs.VectorFrameSequence(u, dx=dx, scale=scale)

        return self._displacement

    def get_velocities(
        self,
        unit: str = "um",
        scale: float = 1.0,
        spacing: int = 1,
    ):
        assert unit in ["pixels", "um"]
        data = self.data

        if scale > 1.0:
            raise ValueError("Cannot have scale larger than 1.0")

        scaled_data = data
        if scale < 1.0:
            scaled_data = scaling.resize_data(data, scale)

        v = self._get_velocities(
            scaled_data.frames, scaled_data.time_stamps, spacing=spacing, **self.options
        )
        dx = 1

        scale *= self.data_scale

        v /= scale

        if unit == "um":
            v *= scaled_data.info.get("um_per_pixel", 1.0)
            dx *= scaled_data.info.get("um_per_pixel", 1.0)
        else:
            v /= scale

        if not isinstance(v, da.Array):
            v = da.from_array(v)

        self._velocity = fs.VectorFrameSequence(v, dx=dx, scale=scale)

        return self._velocity

    @property
    def reference_frame(self) -> str:
        return self._reference_frame

    @property
    def reference_frame_index(self) -> Optional[int]:
        return self._reference_frame_index

    @property
    def reference_image(self) -> np.ndarray:
        return self._reference_image

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"data={self.data}, flow_algorithm={self.flow_algorithm}, reference_frame={self._reference_frame})"
        )
