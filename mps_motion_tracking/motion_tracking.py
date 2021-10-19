import logging
from enum import Enum
from typing import Optional
from typing import Tuple
from typing import Union

import dask.array as da
import numpy as np

from . import block_matching
from . import dualtvl10
from . import farneback
from . import frame_sequence as fs
from . import lucas_kanade
from . import scaling
from . import utils
from .mechanics import compute_velocity

logger = logging.getLogger(__name__)


class FLOW_ALGORITHMS(str, Enum):
    farneback = "farneback"
    dualtvl10 = "dualtvl10"
    lucas_kanade = "lucas_kanade"
    block_matching = "block_matching"


def _check_algorithm(alg):
    msg = f"Expected flow algorithm to be one of {FLOW_ALGORITHMS._member_names_}, got {alg}"
    if alg not in FLOW_ALGORITHMS._member_names_:
        raise ValueError(msg)


def get_referenece_image(
    reference_frame,
    frames,
    time_stamps: Optional[np.ndarray] = None,
) -> Tuple[str, np.ndarray, int]:

    reference_frame_index = 0
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
            reference_frame_index = next(
                i for i, t in enumerate(time_stamps) if t >= reference_time
            )
        except StopIteration:
            reference_frame_index = len(time_stamps) - 1

        reference_frame_index = int(min(reference_frame_index, len(time_stamps) - 1))
        # Pick neighbouring index
        if reference_frame_index == 0:
            reference_image = frames[
                :, :, reference_frame_index : reference_frame_index + 3
            ].mean(-1)
        elif reference_frame_index == len(time_stamps) - 1:
            reference_image = frames[
                :, :, reference_frame_index - 2 : reference_frame_index + 1
            ].mean(-1)
        else:
            reference_image = frames[
                :, :, reference_frame_index - 1 : reference_frame_index + 2
            ].mean(-1)

    return reference_str, reference_image, reference_frame_index


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

        (
            self._reference_frame,
            self._reference_image,
            self._reference_frame_index,
        ) = get_referenece_image(reference_frame, data.frames, data.time_stamps)

        self._handle_algorithm(options)

    def _handle_algorithm(self, options):
        _check_algorithm(self.flow_algorithm)

        if self.flow_algorithm == "lucas_kanade":
            self._get_displacements = lucas_kanade.get_displacements
            self.options = lucas_kanade.default_options()

        elif self.flow_algorithm == "block_matching":
            self._get_displacements = block_matching.get_displacements
            self.options = block_matching.default_options()

        elif self.flow_algorithm == "farneback":
            self._get_displacements = farneback.get_displacements
            self.options = farneback.default_options()

        elif self.flow_algorithm == "dualtvl10":
            self._get_displacements = dualtvl10.get_displacements
            self.options = dualtvl10.default_options()

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
            u /= scale
            if unit == "um":
                u *= scaled_data.info.get("um_per_pixel", 1.0)
                dx *= scaled_data.info.get("um_per_pixel", 1.0)

            if not isinstance(u, da.Array):
                u = da.from_array(u)

            u = da.swapaxes(u, 2, 3)
            self._displacement = fs.VectorFrameSequence(u, dx=dx, scale=scale)

        return self._displacement

    def get_velocities(
        self,
        recompute: bool = False,
        unit: str = "um",
        scale: float = 1.0,
    ):
        u = self.get_displacements(recompute=recompute, unit=unit, scale=scale)
        return compute_velocity(u.array, self.data.time_stamps)

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
