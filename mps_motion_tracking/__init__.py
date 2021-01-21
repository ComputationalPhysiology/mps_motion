"""Top-level package for MPS Motion Tracking."""

__author__ = """Henrik Finsberg"""
__email__ = "henriknf@simula.no"
__version__ = "0.1.0"

import logging as _logging

from . import dualtvl10, farneback, lucas_kanade, mechanics, utils
from .mechanics import Mechancis

_logging.basicConfig(level=_logging.INFO)

__all__ = ["farneback", "dualtvl10", "lucas_kanade", "utils", "mechanics", "Mechancis"]
