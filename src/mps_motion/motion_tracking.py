import logging
from enum import Enum
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import ap_features as apf
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


class ReferenceFrameError(RuntimeError):
    pass


def _check_algorithm(alg):
    msg = f"Expected flow algorithm to be one of {FLOW_ALGORITHMS._member_names_}, got {alg}"
    if alg not in FLOW_ALGORITHMS._member_names_:
        raise ValueError(msg)


def estimate_referece_image_from_velocity(
    t: np.ndarray,
    v: np.ndarray,
    rel_tol=0.01,
    raise_on_failure: bool = False,
) -> int:

    # Estimate baseline
    background = apf.background.background(x=t, y=v)

    # Find points close to the baseline
    inds = np.isclose(v, background, atol=np.abs(v.max() - v.min()) * rel_tol)

    msg = (
        "Unable to find any values are the baseline. "
        "Please try a smaller tolerance when estimating the reference image"
    )
    if not inds.any():
        # No values are this close to the baseline
        logger.warning(msg)

        if raise_on_failure:
            raise ReferenceFrameError(msg)

        # If not raising, just return the index of the first frame
        return 0

    # Find biggest connected segment on the baseline
    from collections import Counter

    groups = np.diff(inds).cumsum()
    counts = Counter(groups).most_common()
    group_index = 0
    ref_index_at_baseline = False

    while not ref_index_at_baseline:
        try:
            index = counts[group_index][0]
        except IndexError:
            logger.warning(msg)
            if raise_on_failure:
                raise ReferenceFrameError(msg)
            # Just return the index of the first frame
            return 0

        # Choose the value at the center of this segment
        ref_index = int(np.median(np.where(groups == index)[0]))
        ref_index_at_baseline = inds[ref_index]
        group_index += 1

    return ref_index


def get_reference_image(
    reference_frame: Union[float, str, RefFrames],
    frames: np.ndarray,
    time_stamps: Optional[np.ndarray] = None,
    smooth_ref_transition: bool = True,
) -> np.ndarray:
    """Get reference frame

    Parameters
    ----------
    reference_frame : Union[float, str, RefFrames]
        Either a float of string representing the
        time-point of the frame that should be used as reference
    frames : np.ndarray
        The image frames
    time_stamps : Optional[np.ndarray], optional
        The time stamps, by default None
    smooth_ref_transition : bool, optional
        If true, compute the mean frame of the three closest frames, by default True

    Returns
    -------
    np.ndarray
        reference_image

    """
    try:
        reference_time = float(reference_frame)

    except ValueError:
        refs = RefFrames._member_names_
        msg = (
            "Expected reference frame to be an integer or one of "
            f"{refs}, got {reference_frame}"
        )
        if str(reference_frame) not in refs:
            raise ValueError(msg)
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

    return reference_image


class OpticalFlow:
    def __init__(
        self,
        data: utils.MPSData,
        flow_algorithm: FLOW_ALGORITHMS = FLOW_ALGORITHMS.farneback,
        filter_options: Optional[Dict[str, Any]] = None,
        data_scale: float = 1.0,
        **options,
    ):
        self.data = data

        self.flow_algorithm = flow_algorithm

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
        reference_frame: Union[float, str, RefFrames] = 0,
        smooth_ref_transition: bool = True,
        reference_image: Optional[np.ndarray] = None,
    ) -> fs.VectorFrameSequence:
        """Compute motion of all images relative to reference frame

        Parameters
        ----------
        recompute : bool, optional
            If already computed set this to true if you want to
            recomputed, by default False
        unit : str, optional
            Either 'pixels' or 'um', by default "pixels".
            If using 'um' them the MPSData.info has to contain the
            key 'um_per_pixel'.
        scale : float, optional
            If less than 1.0, down-sample images before estimating motion, by default 1.0
        reference_frame: float, str, RefFrames, optional
            A float or string indicating the reference frame to use. If the value
            is a float, it should refer to the time-point of the reference frame to use.
            If you known the exact index to use, then you can use the `reference_image` parameter,
            by default 0
        smooth_ref_transition : bool, optional
            If true, compute the mean frame of the three closest frames, by default True
        reference_image: np.ndarray, optional
            The reference image to use containing the actual pixel values.
            Note that if you provide an argument for this, then the `reference_frame`
            parameter is not used, by default None

        Returns
        -------
        np.ndarray
            The displacements
        """
        assert unit in ["pixels", "um"]

        if scale > 1.0:
            raise ValueError("Cannot have scale larger than 1.0")

        if scale < 1.0:
            data = scaling.resize_data(self.data, scale)
        else:
            data = self.data

        if reference_image is None:
            reference_image = get_reference_image(
                reference_frame,
                data.frames,
                data.time_stamps,
                smooth_ref_transition=smooth_ref_transition,
            )

        if not hasattr(self, "_displacement") or recompute:

            u = self._get_displacements(data.frames, reference_image, **self.options)
            dx = 1

            scale *= self.data_scale

            u /= scale

            if unit == "um":
                u *= data.info.get("um_per_pixel", 1.0)
                dx *= data.info.get("um_per_pixel", 1.0)
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

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"data={self.data}, flow_algorithm={self.flow_algorithm})"
        )
