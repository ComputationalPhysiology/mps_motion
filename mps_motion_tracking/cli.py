__doc__ = """Motion tracking of MPS data

This is software to estimate motion in Brightfield images.

"""
import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml

from . import Mechanics, OpticalFlow
from . import motion_tracking as mt
from . import utils

logger = logging.getLogger(__name__)


def print_dict(d: Dict[str, Any], fmt="{:<10}: {}"):
    s = ""
    for k, v in d.items():
        s += fmt.format(k, str(v)) + "\n"
    return s


def normalize_baseline_func(
    arr: np.ndarray,
    normalize_baseline: bool,
    index: Optional[int],
):
    if normalize_baseline:
        assert index is not None
        return arr - arr[index]
    return arr


def plot_displacement(
    mechanics,
    time_stamps,
    path,
    normalize_baseline: bool,
    index: Optional[int],
):

    fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    ax[0].plot(
        time_stamps,
        normalize_baseline_func(
            mechanics.u.norm().mean().compute(),
            normalize_baseline,
            index,
        ),
    )
    ax[1].plot(
        time_stamps,
        normalize_baseline_func(
            mechanics.u.x.mean().compute(),
            normalize_baseline,
            index,
        ),
    )
    ax[2].plot(
        time_stamps,
        normalize_baseline_func(
            mechanics.u.y.mean().compute(),
            normalize_baseline,
            index,
        ),
    )

    for axi in ax:
        axi.grid()
        axi.set_ylabel("Displacement (\u00B5m)")
    ax[0].set_title("Norm")
    ax[1].set_title("X")
    ax[2].set_title("Y")
    ax[2].set_xlabel("Time (ms)")
    fig.savefig(path)


def plot_strain(
    mechanics,
    time_stamps,
    path,
    normalize_baseline: bool,
    index: Optional[int],
):
    fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    E = mechanics.E
    Exx = E.x.mean()
    Exy = E.xy.mean()
    Eyy = E.y.mean()
    ax[0].plot(time_stamps, normalize_baseline_func(Exx, normalize_baseline, index))
    ax[1].plot(time_stamps, normalize_baseline_func(Exy, normalize_baseline, index))
    ax[2].plot(time_stamps, normalize_baseline_func(Eyy, normalize_baseline, index))

    for axi in ax.flatten():
        axi.grid()

    ax[0].set_ylabel("$E_{xx}$")
    ax[1].set_ylabel("$E_{xy}$")
    ax[2].set_ylabel("$E_{yy}$")
    ax[2].set_xlabel("Time (ms)")
    fig.savefig(path)


def load_data(filename_):
    try:
        import mps

        data = mps.MPS(filename_)
    except ImportError:
        logger.warning("Missing `mps` pacakge.")
        from .utils import MPSData

        data = MPSData(**np.load(filename_, allow_pickle=True).item())
    return data


def main(
    filename: str,
    algorithm: mt.FLOW_ALGORITHMS = mt.FLOW_ALGORITHMS.farneback,
    outdir: Optional[str] = None,
    reference_frame: str = "0",
    normalize_baseline: bool = False,
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
    else:
        outdir_ = Path(outdir)

    settings = {
        "filename": filename_,
        "algorithm": algorithm,
        "outdir": outdir_,
        "scale": scale,
        "reference_frame": reference_frame,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    logger.debug("\nSettings : \n{}".format(print_dict(settings)))

    settings_file = outdir_.joinpath("settings.yaml")
    disp_file = outdir_.joinpath("displacement.npy")
    # Check if file is allready analyzed
    opt_flow = None
    index = None
    if settings_file.is_file() and disp_file.is_file() and not overwrite:
        with open(settings_file, "r") as f:
            settings = yaml.load(f, Loader=yaml.SafeLoader)
        disp = np.load(disp_file)

    else:
        if not filename_.is_file():
            raise IOError(f"File {filename_} does not exist")

        if not (0 < scale <= 1.0):
            raise ValueError("Scale has to be between 0 and 1.0")

        data = load_data(filename_)
        logger.info(f"Analyze motion in file {filename}...")
        opt_flow = OpticalFlow(data, algorithm, reference_frame=reference_frame)
        disp = opt_flow.get_displacements(scale=scale)

    if normalize_baseline:
        if opt_flow is None:
            data = load_data(filename)
            opt_flow = OpticalFlow(data, algorithm, reference_frame=reference_frame)

        if opt_flow.reference_frame_index is None:
            print(
                "Unable to normalize baseline. Please select a different reference frame",
            )
        else:
            index = opt_flow.reference_frame_index

    mech = Mechanics(disp)
    outdir_.mkdir(exist_ok=True, parents=True)
    # Plot
    plot_displacement(
        mech,
        data.time_stamps,
        outdir_.joinpath("displacement.png"),
        normalize_baseline=normalize_baseline,
        index=index,
    )
    plot_strain(
        mech,
        data.time_stamps,
        outdir_.joinpath("strain.png"),
        normalize_baseline=normalize_baseline,
        index=index,
    )

    with open(settings_file, "w") as f:
        yaml.dump(settings, f)

    np.save(disp_file, disp.array.compute())
