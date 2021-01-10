"""Top-level package for MPS Motion Tracking."""

__author__ = """Henrik Finsberg"""
__email__ = "henriknf@simula.no"
__version__ = "0.1.0"

from . import block_matching, farneback, motion_tracking, test_data
from .motion_tracking import MotionTracking

__all__ = [
    "motion_tracking",
    "MotionTracking",
    "test_data",
    "block_matching",
    "farneback",
]
