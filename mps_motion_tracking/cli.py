__doc__ = """Motion tracking of MPS data

This is software to estimate motion in Brightfield images.

"""
import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from . import OpticalFlow
from . import motion_tracking as mt
from . import utils

logger = logging.getLogger(__name__)


def print_dict(d: Dict[str, Any], fmt="{:<10}: {}"):
    s = ""
    for k, v in d.items():
        s += fmt.format(k, str(v)) + "\n"
    return s


def main(
    filename: str,
    algoritm: mt.FLOW_ALGORITHMS = mt.FLOW_ALGORITHMS.farneback,
    outdir: Optional[str] = None,
    scale: float = 0.3,
    verbose: bool = False,
    overwrite: bool = True,
):
    """
    Estimate motion in stack of images

    \b
    Parameters
    ----------
    filename : str
        Path to file to be analyzed, typically an .nd2 or .czi Brightfield file
    algoritm : mt.FLOW_ALGORITHMS, optional
        The algorithm used to estimate motion, by default mt.FLOW_ALGORITHMS.farneback
    outdir : Optional[str], optional
        Directory where to store the results. If not provided, a folder with the the same
        as the filename will be created and results will be stored in a subfolder inside
        that called `motion`.
    scale : float, optional
        Rescale data before running motion track. This is useful if the spatial resoltion
        of the images are large. Scale = 1.0 will keep the original size, by default 0.3
    verbose : bool, optional
        Print more to the console, by default False
    overwrite : bool, optional
        If `outdir` allready exist an contains the relevant files then set this to false to
        use that data, by default True

    \b
    Raises
    ------
    IOError
        If filename does not exist
    ValueError
        If scale is zero or lower, or higher than 1.0
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
        utils.logger.setLevel(logging.DEBUG)
        mt.logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        utils.logger.setLevel(logging.INFO)
        mt.logger.setLevel(logging.INFO)

    filename_ = Path(filename)
    if outdir is None:
        outdir_ = filename_.with_suffix("").joinpath("motion")
        outdir_.mkdir(exist_ok=True, parents=True)
    else:
        outdir_ = Path(outdir)

    settings = {
        "filename": filename_,
        "algorithm": algoritm,
        "outdir": outdir_,
        "scale": scale,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    logger.debug("\nSettings : \n{}".format(print_dict(settings)))

    settings_file = outdir_.joinpath("settings.yaml")
    disp_file = outdir_.joinpath("displacement.npy")
    # Check if file is allready analyzed
    if settings_file.is_file() and disp_file.is_file() and not overwrite:
        with open(settings_file, "r") as f:
            settings = yaml.load(f)
        disp = np.load(disp_file)

    else:
        if not filename_.is_file():
            raise IOError(f"File {filename_} does not exist")

        if not (0 < scale <= 1.0):
            raise ValueError("Scale has to be between 0 and 1.0")

        try:
            import mps

            data = mps.MPS(filename_)
        except ImportError:
            logger.warning("Missing `mps` pacakge.")
            from .motion_tracking.utils import MPSData

            data = MPSData(**np.load(filename_, allow_pickle=True).item())

        logger.info(f"Analyze motion in file {filename}...")
        data = utils.resize_data(data=data, scale=scale)
        opt_flow = OpticalFlow(data, algoritm)
        disp = opt_flow.get_displacements()

    # mech = Mechancis(disp)

    with open(settings_file, "w") as f:
        yaml.dump(settings, f)
    np.save(disp_file, disp)
