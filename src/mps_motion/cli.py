__doc__ = """
Motion tracking of MPS data
This is software to estimate motion in Brightfield images.
"""
import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import ap_features as apf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import mps

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
        if "average_time" in name:
            continue
        if "average_trace" in name:
            time = results[f"{name[0]}_average_time"]
        else:
            time = results["time"]

        fig, ax = plt.subplots()
        ax.plot(time[: len(arr)], arr)
        ax.set_xlabel(f"Time [{time_unit}]")
        if name.startswith("u"):
            ylabel = f"Displacement {name.split('_')[-1]} [\u00b5m]"
        else:
            ylabel = f"Velocity {name.split('_')[-1]} [\u00b5m/s]"
        ax.set_ylabel(ylabel)
        ax.grid()
        fig.savefig(outdir.joinpath(name).with_suffix(".png"))


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return obj.as_posix()
        elif isinstance(mt.FLOW_ALGORITHMS):
            return str(obj)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def analyze_motion_array(y: np.ndarray, t=np.ndarray, intervals=None, background_correction_method="subtract"):
    trace = apf.Beats(
        y=y,
        t=t,
        background_correction_method=background_correction_method,
        background_correction_kernel=10,
        chopping_options={
            "threshold_factor": 0.3,
        },
        intervals=intervals,
    )
    try:
        beats = trace.beats
        if len(beats) == 0:
            raise apf.chopping.EmptyChoppingError
    except (apf.chopping.InvalidChoppingError, apf.chopping.EmptyChoppingError):
        chopped = dict(y=[trace.y.tolist()], t=[trace.t.tolist()])
        chopped_aligned = dict(y=[trace.y.tolist()], t=[trace.t.tolist()])
        avg_trace = trace.y.tolist()
        avg_time = trace.t.tolist()
        intervals = None
    else:
        chopped = dict(
            y=[beat.y.tolist() for beat in beats],
            t=[beat.t.tolist() for beat in beats],
        )
        try:
            chopped_aligned_beats = apf.beat.align_beats(beats)
        except Exception:
            chopped_aligned_beats = beats
        chopped_aligned = dict(
            y=[beat.y.tolist() for beat in chopped_aligned_beats],
            t=[beat.t.tolist() for beat in chopped_aligned_beats],
        )
        avg_beat = trace.average_beat()

        avg_trace = avg_beat.y.tolist()
        avg_time = avg_beat.t.tolist()
        intervals = trace.intervals

    original = None if np.isnan(trace.original_y).any() else trace.original_y.tolist()
    corrected = None if np.isnan(trace.y).any() else trace.y.tolist()
    average_trace = None if np.isnan(avg_trace).any() else avg_trace
    average_time = None if np.isnan(avg_time).any() else avg_time
    background = None if np.isnan(trace.background).any() else trace.background.tolist()
    chopped_ = None if np.any([np.isnan(yi).any() for yi in chopped["y"]]) else chopped
    chopped_aligned_ = None if np.any([np.isnan(yi).any() for yi in chopped_aligned["y"]]) else chopped_aligned
    return {
        "original": original,
        "corrected": corrected,
        "average_trace": average_trace,
        "average_time": average_time,
        "background": background,
        "chopped": chopped_,
        "chopped_aligned": chopped_aligned_,
    }, intervals


def plot_selection(
    fname: Union[str, Path],
    frame: np.ndarray,
    start_x: int,
    end_x: int,
    start_y: int,
    end_y: int,
) -> None:
    fig, ax = plt.subplots()
    ax.imshow(frame)
    rect = patches.Rectangle(
        (start_x, start_y),
        end_x - start_x,
        end_y - start_y,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    # Add the patch to the Axes
    ax.add_patch(rect)
    fig.savefig(fname)


def main(  # noqa: C901
    filename: Union[str, Path],
    algorithm: mt.FLOW_ALGORITHMS = mt.FLOW_ALGORITHMS.farneback,
    outdir: Optional[str] = None,
    reference_frame: str = "0",
    estimate_reference_frame: bool = True,
    scale: float = 1.0,
    apply_filter: bool = True,
    spacing: int = 5,
    compute_xy_components: bool = False,
    make_displacement_video: bool = True,
    make_velocity_video: bool = True,
    verbose: bool = False,
    video_disp_scale: int = 4,
    video_disp_step: int = 24,
    video_vel_scale: int = 1,
    video_vel_step: int = 24,
    suppress_error: bool = False,
    start_x: Optional[int] = None,
    end_x: Optional[int] = None,
    start_y: Optional[int] = None,
    end_y: Optional[int] = None,
    start_t: Optional[float] = None,
    end_t: Optional[float] = None,
    background_correction_method: str = "subtract",
):
    """
    Estimate motion in stack of images

    Parameters
    ----------
    filename : str
        Path to file to be analyzed, typically an .nd2 or .czi Brightfield file
    algorithm : mt.FLOW_ALGORITHMS, optional
        The algorithm used to estimate motion, by default mt.FLOW_ALGORITHMS.farneback
    outdir : Optional[str], optional
        Directory where to store the results. If not provided, a folder with the the same
        as the filename will be created and results will be stored in a subfolder inside
        that called `motion`.
    reference_frame: str, optional
        Which frame should be the reference frame when computing the displacements.
        This can either be a number indicating the timepoint,
        or the value 'mean', 'median', 'max' or 'mean'. Default: '0' (i.e the first frame)
    estimate_reference_frame : bool, optional
        If True, estimate the the reference frame, by default True. Note that this will overwrite
        the argument `reference_frame`
    scale : float, optional
        Rescale data before running motion track. This is useful if the spatial resolution
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
    video_disp_scale : int, optional
        Scale of the displacement vectors in the video, by default 4
    video_disp_step : int, optional
        Step between frames in the displacement video, by default 24
    video_vel_scale : int, optional
        Scale of the velocity vectors in the video, by default 1
    video_vel_step : int, optional
        Step between frames in the velocity video, by default 24
    suppress_error : bool, optional
        Suppress error if file does not exist, by default False
    start_x : Optional[int], optional
        Start x coordinate, by default None
    end_x : Optional[int], optional
        End x coordinate, by default None
    start_y : Optional[int], optional
        Start y coordinate, by default None
    end_y : Optional[int], optional
        End y coordinate, by default None
    start_t : Optional[float], optional
        Start time, by default None
    end_t : Optional[float], optional
        End time, by default None
    background_correction_method : str, optional
        Method to use for background correction, by default "subtract".
        Choose between 'subtract' and 'none'

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

    if background_correction_method not in ["subtract", "none"]:
        raise ValueError("background_correction_method has to be 'subtract' or 'none'")

    filename_ = Path(filename)
    if filename_.is_dir():
        # Traverse recursively
        for f in filename_.iterdir():
            main(
                filename=f,
                algorithm=algorithm,
                reference_frame=reference_frame,
                estimate_reference_frame=estimate_reference_frame,
                outdir=outdir,
                scale=scale,
                apply_filter=apply_filter,
                spacing=spacing,
                compute_xy_components=compute_xy_components,
                make_displacement_video=make_displacement_video,
                make_velocity_video=make_velocity_video,
                verbose=verbose,
                video_disp_scale=video_disp_scale,
                video_disp_step=video_disp_step,
                video_vel_scale=video_vel_scale,
                video_vel_step=video_vel_step,
                suppress_error=True,
                start_x=start_x,
                end_x=end_x,
                start_y=start_y,
                end_y=end_y,
                start_t=start_t,
                end_t=end_t,
            )

    if filename_.suffix not in mps.load.valid_extensions + [".npy"]:
        return
    try:
        data = mps.MPS(filename_)
    except Exception:
        if suppress_error:
            return
        raise
    logger.info(f"Analyze {filename_.absolute()}")

    if outdir is None:
        outdir_ = filename_.with_suffix("").joinpath("motion")
    else:
        outdir_ = Path(outdir)

    logger.info(f"Saving output to {outdir_.absolute()}")

    settings = {
        "filename": filename_,
        "algorithm": algorithm,
        "outdir": outdir_,
        "scale": scale,
        "reference_frame": reference_frame,
        "estimate_reference_frame": estimate_reference_frame,
        "start_x": start_x,
        "end_x": end_x,
        "start_y": start_y,
        "end_y": end_y,
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

    logger.info(f"Analyze motion in file {filename}...")
    Nx, Ny, Nt = data.frames.shape
    original_frame = data.frames[:, :, 0].T
    start_x = start_x or 0
    end_x = end_x or Nx
    start_y = start_y or 0
    end_y = end_y or Ny

    def find_time_index(t_star, default):
        index = default
        if t_star is None:
            return index
        try:
            index = next(i for i, t in enumerate(data.time_stamps) if t >= t_star)
        except StopIteration:
            pass
        return index

    t0 = find_time_index(start_t, default=0)
    t1 = find_time_index(end_t, default=Nt)

    data = utils.MPSData(
        frames=data.frames[start_x:end_x, start_y:end_y, t0:t1],
        time_stamps=data.time_stamps[t0:t1],
        info=data.info,
        metadata=data.metadata,
    )

    if scale < 1.0:
        data = scaling.resize_data(data, scale=scale)

    opt_flow = OpticalFlow(
        data,
        flow_algorithm=algorithm,
    )

    if estimate_reference_frame:
        logger.info("Estimating reference frame")
        v = opt_flow.get_velocities(spacing=5)
        v_norm = v.norm().mean().compute()
        reference_frame_index = mt.estimate_referece_image_from_velocity(
            t=data.time_stamps[:-5],
            v=v_norm,
        )
        reference_frame = data.time_stamps[reference_frame_index]
        logger.info(
            f"Found reference frame at index {reference_frame_index} and time {reference_frame:.2f}",
        )
    else:
        v = None

    u = opt_flow.get_displacements(reference_frame=reference_frame)
    factor = 1000.0 if data.info["time_unit"] == "ms" else 1.0
    v = Mechanics(u, t=data.time_stamps / factor).velocity(spacing=spacing)
    if apply_filter:
        logger.info("Apply filter")
        u_norm_max = u.norm().max().compute()
        mask = u_norm_max < u_norm_max.mean()
        fig, ax = plt.subplots()
        ax.imshow(mask.T)
        fig.savefig(outdir_.joinpath("mask.png"))
        v.apply_mask(mask)
        u.apply_mask(mask)

    results = {"time": data.time_stamps}
    logger.info("Compute displacement norm")
    results["u_original"] = u.norm().mean().compute()
    logger.info("Compute velocity norm")
    results["v_original"] = v.norm().mean().compute()

    if compute_xy_components:
        logger.info("Compute displacement x component")
        results["u_x"] = u.x.mean().compute()
        logger.info("Compute displacement y component")
        results["u_y"] = u.y.mean().compute()
        logger.info("Compute velocity x component")
        results["v_x"] = v.x.mean().compute()
        logger.info("Compute velocity y component")
        results["v_y"] = v.y.mean().compute()

    logger.info("Compute average")
    u_data, intervals = analyze_motion_array(
        y=results["u_original"],
        t=data.time_stamps,
        background_correction_method=background_correction_method,
    )
    v_data, _ = analyze_motion_array(
        y=results["v_original"],
        t=data.time_stamps[:-spacing],
        intervals=intervals,
        background_correction_method=background_correction_method,
    )
    for key in ["corrected", "average_trace", "average_time"]:
        results[f"u_{key}"] = u_data[key]
        results[f"v_{key}"] = v_data[key]

    outdir_.mkdir(exist_ok=True, parents=True)
    plot_selection(
        fname=outdir_ / "selection.png",
        frame=original_frame,
        start_x=start_x,
        start_y=start_y,
        end_x=end_x,
        end_y=end_y,
    )
    plot_traces(results=results, outdir=outdir_, time_unit=data.info["time_unit"])

    logger.info("Compute features")
    features = stats.compute_features(
        u=results["u_original"],
        v=results["v_original"],
        t=data.time_stamps,
    )

    analysis = stats.analysis_from_arrays(
        u=results["u_original"],
        v=results["v_original"],
        t=data.time_stamps,
    )

    for key in [
        "u_peaks",
        "u_peaks_first",
        "u_width50_global",
        "u_width50_first",
        "u_width50_global_first_last",
        "u_width50_first_first_last",
        "time_above_half_height",
        "time_above_half_height_first",
        "time_between_contraction_and_relaxation",
        "max_contraction_velocity",
        "max_relaxation_velocity",
    ]:
        features[key] = getattr(analysis, key)

    settings_file.write_text(json.dumps(settings, cls=JSONEncoder, indent=2))

    mps.utils.to_csv(results, results_file)
    mps.utils.to_csv(features, features_file)
    (outdir_ / "mean.json").write_text(json.dumps(analysis.mean, indent=2))
    (outdir_ / "std.json").write_text(json.dumps(analysis.std, indent=2))

    if make_displacement_video:
        visu.quiver_video(
            data,
            u,
            outdir_.joinpath("displacement_movie"),
            step=video_disp_step,
            vector_scale=video_disp_scale,
        )

    if make_velocity_video:
        visu.quiver_video(
            data,
            v,
            outdir_.joinpath("velocity_movie"),
            step=video_vel_step,
            vector_scale=video_vel_scale,
            offset=spacing,
        )
