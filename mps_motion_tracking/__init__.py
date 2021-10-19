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
    mechanics,
    motion_tracking,
    scaling,
    utils,
    visu,
)
from .mechanics import Mechancis
from .motion_tracking import FLOW_ALGORITHMS, OpticalFlow


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
    "Mechancis",
    "motion_tracking",
    "FLOW_ALGORITHMS",
    "OpticalFlow",
    "scaling",
    "visu",
]
