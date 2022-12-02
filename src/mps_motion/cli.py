__doc__ = """
Motion tracking of MPS data
This is software to estimate motion in Brightfield images.
"""
import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import json

from . import Mechanics, OpticalFlow
from . import motion_tracking as mt
from . import utils
from . import stats
from . import visu
from . import scaling

logger = logging.getLogger(__name__)


def print_dict(d: Dict[str, Any], fmt="{:<10}: {}"):
    s = ""
    for k, v in d.items():
        s += fmt.format(k, str(v)) + "\n"
    return s


def plot_traces(results, outdir: Path, time_unit="ms"):
    assert "time" in results, "Missing time array"
    for name, arr in results.items():
        if name == "time":
            continue

        fig, ax = plt.subplots()
        ax.plot(results["time"][: len(arr)], arr)
        ax.set_xlabel(f"Time [{time_unit}]")
        if name.startswith("u"):
            ylabel = f"Displacement {name.split('_')[-1]} [\u00B5m]"
        else:
            ylabel = f"Velocity {name.split('_')[-1]} [\u00B5m/s]"
        ax.set_ylabel(ylabel)
        ax.grid()
        fig.savefig(outdir.joinpath(name).with_suffix(".png"))


def load_data(filename_):
    try:
        import mps

        data = mps.MPS(filename_)
    except ImportError as e:
        raise ImportError(
            (
                "Missing `mps` pacakge. Please install it. "
                "'python -m pip install cardiac-mps'"
            ),
        ) from e
    return data


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return obj.as_posix()
        elif isinstance(mt.FLOW_ALGORITHMS):
            return str(obj)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def main(
    filename: str,
    algorithm: mt.FLOW_ALGORITHMS = mt.FLOW_ALGORITHMS.farneback,
    outdir: Optional[str] = None,
    reference_frame: str = "0",
    scale: float = 1.0,
    apply_filter: bool = True,
    spacing: int = 5,
    compute_xy_components: bool = False,
    make_displacement_video: bool = True,
    make_velocity_video: bool = True,
    verbose: bool = False,
):
    """
    Estimate motion in stack of images

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
    apply_filter, bool, optional
        If True, set pixels with max displacement lower than the mean maximum displacement
        to zero. This will prevent non-tissue pixels to be included, which is especially
        error prone for velocity estimations, by default True.
    spacing: int, optional
        Spacing between frames in velocity computations, by default 5.
    compute_xy_components: bool, optional
        If True the compute x- and y components of the displacement and
        velocity and plot them as well, by default False.
    make_displacement_video: bool, optional
        If True, create video of displacement vectors, by default False.
    make_velocity_video: bool, optional
        If True, create video of velocity vectors, by default False.
    verbose : bool, optional
        Print more to the console, by default False
    overwrite : bool, optional
        If `outdir` allready exist an contains the relevant files then set this to false to
        use that data, by default True

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

    settings_file = outdir_.joinpath("settings.json")
    results_file = outdir_.joinpath("results.csv")
    features_file = outdir_.joinpath("features.csv")

    if not filename_.is_file():
        raise IOError(f"File {filename_} does not exist")

    if not (0 < scale <= 1.0):
        raise ValueError("Scale has to be between 0 and 1.0")

    data = load_data(filename_)
    logger.info(f"Analyze motion in file {filename}...")
    if scale < 1.0:
        data = scaling.resize_data(data, scale=scale)
    opt_flow = OpticalFlow(
        data,
        flow_algorithm=algorithm,
    )
    u = opt_flow.get_displacements(reference_frame=reference_frame)
    factor = 1000.0 if data.info["time_unit"] == "ms" else 1.0
    v = Mechanics(u, t=data.time_stamps / factor).velocity(spacing=spacing)
    if apply_filter:
        logger.info("Apply filter")
        u_norm_max = u.norm().max().compute()
        mask = u_norm_max < u_norm_max.mean()
        v.apply_mask(mask)
        u.apply_mask(mask)

    results = {"time": data.time_stamps}
    logger.info("Compute displacement norm")
    results["u_norm"] = u.norm().mean().compute()
    logger.info("Compute velocity norm")
    results["v_norm"] = v.norm().mean().compute()

    if compute_xy_components:
        logger.info("Compute displacement x component")
        results["u_x"] = u.x.mean().compute()
        logger.info("Compute displacement y component")
        results["u_y"] = u.y.mean().compute()
        logger.info("Compute velocity x component")
        results["v_x"] = v.x.mean().compute()
        logger.info("Compute velocity y component")
        results["v_y"] = v.y.mean().compute()

    outdir_.mkdir(exist_ok=True, parents=True)
    plot_traces(results=results, outdir=outdir_, time_unit=data.info["time_unit"])

    logger.info("Compute features")
    features = stats.compute_features(
        u=results["u_norm"],
        v=results["v_norm"],
        t=data.time_stamps,
    )
    with open(settings_file, "w") as f:
        json.dump(settings, f, cls=JSONEncoder, indent=2)

    import mps

    mps.utils.to_csv(results, results_file)
    mps.utils.to_csv(features, features_file)

    if make_displacement_video:
        visu.quiver_video(
            data,
            u,
            outdir_.joinpath("displacement_movie"),
            step=24,
            vector_scale=4,
        )

    if make_velocity_video:
        visu.quiver_video(
            data,
            v,
            outdir_.joinpath("velocity_movie"),
            step=24,
            vector_scale=1,
            offset=spacing,
        )
