"""Top-level package for MPS Motion Tracking."""

__author__ = """Henrik Finsberg"""
__email__ = "henriknf@simula.no"
__version__ = "0.1.0"

import logging as _logging

from . import (
    block_matching,
    dualtvl10,
    farneback,
    lucas_kanade,
    mechanics,
    motion_tracking,
    utils,
)
from .mechanics import Mechancis
from .motion_tracking import (
    DENSE_FLOW_ALGORITHMS,
    SPARSE_FLOW_ALGORITHMS,
    DenseOpticalFlow,
    SparseOpticalFlow,
)

_logging.basicConfig(level=_logging.INFO)

__all__ = [
    "farneback",
    "dualtvl10",
    "lucas_kanade",
    "block_matching",
    "utils",
    "mechanics",
    "Mechancis",
    "motion_tracking",
    "DenseOpticalFlow",
    "SparseOpticalFlow",
    "DENSE_FLOW_ALGORITHMS",
    "SPARSE_FLOW_ALGORITHMS",
]
