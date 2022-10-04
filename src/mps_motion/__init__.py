"""Top-level package for MPS Motion Tracking."""
import logging as _logging
from importlib.metadata import metadata

import daiquiri as _daiquiri

from . import block_matching
from . import dualtvl1
from . import farneback
from . import filters
from . import frame_sequence
from . import lucas_kanade
from . import mechanics
from . import motion_tracking
from . import scaling
from . import stats
from . import utils
from . import visu
from .frame_sequence import FrameSequence
from .frame_sequence import TensorFrameSequence
from .frame_sequence import VectorFrameSequence
from .mechanics import Mechanics
from .motion_tracking import FLOW_ALGORITHMS
from .motion_tracking import list_optical_flow_algorithm
from .motion_tracking import OpticalFlow
from .utils import MPSData

meta = metadata("mps-motion")
__version__ = meta["Version"]
__author__ = meta["Author"]
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]


def set_log_level(level):
    from daiquiri import set_default_log_levels

    loggers = [
        "block_matching.logger",
        "dualtvl1.logger",
        "farneback.logger",
        "lucas_kanade.logger",
        "mechanics.logger",
        "motion_tracking.logger",
        "scaling.logger",
        "utils.logger",
        "visu.logger",
        "frame_sequence.logger",
        "filters.logger",
    ]
    set_default_log_levels((logger, level) for logger in loggers)


_daiquiri.setup(level=_logging.INFO)

__all__ = [
    "farneback",
    "dualtvl1",
    "lucas_kanade",
    "block_matching",
    "utils",
    "mechanics",
    "frame_sequence",
    "Mechanics",
    "MPSData",
    "motion_tracking",
    "FLOW_ALGORITHMS",
    "OpticalFlow",
    "scaling",
    "frame_sequence",
    "visu",
    "filters",
    "FrameSequence",
    "VectorFrameSequence",
    "TensorFrameSequence",
    "stats",
    "list_optical_flow_algorithm",
]
