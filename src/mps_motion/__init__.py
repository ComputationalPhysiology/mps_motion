"""Top-level package for MPS Motion Tracking."""

__author__ = """Henrik Finsberg"""
__email__ = "henriknf@simula.no"
__version__ = "0.1.0"

import logging as _logging
import daiquiri as _daiquiri

from . import (
    block_matching,
    dualtvl1,
    farneback,
    lucas_kanade,
    frame_sequence,
    mechanics,
    motion_tracking,
    scaling,
    utils,
    visu,
    filters,
    stats,
)
from .frame_sequence import FrameSequence, VectorFrameSequence, TensorFrameSequence
from .mechanics import Mechanics
from .motion_tracking import FLOW_ALGORITHMS, OpticalFlow, list_optical_flow_algorithm
from .utils import MPSData


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
