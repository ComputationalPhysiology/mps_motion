import logging
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple

import ap_features as apf
import dask.array as da
import numpy as np
from scipy.signal import find_peaks

from .mechanics import Mechanics
from .utils import Array

logger = logging.getLogger(__name__)


class Analysis(NamedTuple):
    u_peaks: List[float]
    u_peaks_first: List[float]
    u_width50_global: List[float]
    u_width50_first: List[float]
    u_width50_global_first_last: List[float]
    u_width50_first_first_last: List[float]
    time_above_half_height: List[float]
    time_above_half_height_first: List[float]
    time_between_contraction_and_relaxation: List[float]
    max_contraction_velocity: List[float]
    max_relaxation_velocity: List[float]
    frequency: List[float]

    @property
    def mean(self) -> Dict[str, float]:
        return {
            "u_peaks": np.mean(self.u_peaks),
            "u_peaks_first": np.mean(self.u_peaks_first),
            "u_width50_global": np.mean(self.u_width50_global),
            "u_width50_first": np.mean(self.u_width50_first),
            "u_width50_global_first_last": np.mean(self.u_width50_global_first_last),
            "u_width50_first_first_last": np.mean(self.u_width50_first_first_last),
            "time_above_half_height": np.mean(self.time_above_half_height),
            "time_above_half_height_first": np.mean(self.time_above_half_height_first),
            "time_between_contraction_and_relaxation": np.mean(
                self.time_between_contraction_and_relaxation,
            ),
            "max_contraction_velocity": np.mean(self.max_contraction_velocity),
            "max_relaxation_velocity": np.mean(self.max_relaxation_velocity),
            "frequency": np.mean(self.frequency),
        }

    @property
    def std(self) -> Dict[str, float]:
        return {
            "u_peaks": np.std(self.u_peaks),
            "u_width50_global": np.std(self.u_width50_global),
            "u_peaks_first": np.std(self.u_peaks_first),
            "u_width50_first": np.std(self.u_width50_first),
            "u_width50_global_first_last": np.std(self.u_width50_global_first_last),
            "u_width50_first_first_last": np.std(self.u_width50_first_first_last),
            "time_above_half_height": np.std(self.time_above_half_height),
            "time_above_half_height_first": np.std(self.time_above_half_height_first),
            "time_between_contraction_and_relaxation": np.std(
                self.time_between_contraction_and_relaxation,
            ),
            "max_contraction_velocity": np.std(self.max_contraction_velocity),
            "max_relaxation_velocity": np.std(self.max_relaxation_velocity),
            "frequency": np.std(self.frequency),
        }


class ProminenceError(RuntimeError):
    pass


def normalize(y: np.ndarray) -> np.ndarray:
    y = np.array(y)
    return (y - y.min()) / (y.max() - y.min())


def chop_trace(
    u: np.ndarray,
    t: np.ndarray,
    pacing: Optional[np.ndarray] = None,
    background_correction_method="subtract",
    zero_index: Optional[int] = None,
    ignore_pacing: bool = False,
    background_correction_kernel: int = 0,
    threshold_factor: float = 0.5,
    intervals: Optional[List[apf.chopping.Interval]] = None,
) -> Tuple[List[apf.Beat], Optional[List[apf.chopping.Interval]]]:
    trace = apf.Beats(
        u,
        t,
        pacing,
        background_correction_method=background_correction_method,
        background_correction_kernel=background_correction_kernel,
        zero_index=zero_index,
        chopping_options={
            "ignore_pacing": ignore_pacing,
            "threshold_factor": threshold_factor,
        },
        intervals=intervals,
        force_positive=True,
    )

    try:
        beats = trace.beats
        intervals = trace.intervals
    except (apf.chopping.InvalidChoppingError, apf.chopping.EmptyChoppingError):
        beats = [trace.as_beat()]

    return beats, intervals


def compute(u: Array) -> np.ndarray:
    if isinstance(u, da.Array):
        return u.compute()
    return u


def find_two_most_prominent_peaks(y, raise_on_failure: bool = False) -> Tuple[int, int]:
    """Return the indices of the two most promintent peaks

    Parameters
    ----------
    y : np.ndarray
        The signals

    Returns
    -------
    Tuple[int, int]
        Indices of the first and second peak
    """
    from scipy.signal import find_peaks

    peaks: List[int] = []
    prominence = 0.9
    while len(peaks) < 2 and prominence > 0.3:
        peaks, opts = find_peaks(normalize(y), prominence=prominence)
        prominence -= 0.1

    if len(peaks) < 2:
        msg = "Unable to find two most prominent beats"
        logger.warning(msg)
        if raise_on_failure:
            raise ProminenceError(msg)
        t_max = np.argmax(y)
        t2 = int(
            0.5 * (t_max + len(y)),
        )  # Just choose a point between the max and the and
        peaks = [np.argmax(y), t2]

    return (peaks[0], peaks[1])


def analysis_from_mechanics(
    mech: Mechanics,
    pacing: Optional[np.ndarray] = None,
    background_correction_method="subtract",
    zero_index: Optional[int] = None,
    ignore_pacing: bool = False,
    intervals: Optional[List[apf.chopping.Interval]] = None,
) -> Analysis:
    u = compute(mech.u.norm().mean())
    t = mech.t
    v = compute(mech.velocity().norm().mean())
    return analysis_from_arrays(
        u,
        v,
        t,
        pacing=pacing,
        background_correction_method=background_correction_method,
        zero_index=zero_index,
        ignore_pacing=ignore_pacing,
        intervals=intervals,
    )


def find_two_peaks_in_beat(ui: apf.Beat, start_level: float = 0.1) -> np.ndarray:
    level = start_level
    peaks = ui.peaks(prominence_level=level)
    num_peaks = len(peaks)
    if num_peaks == 2:
        dl = 0.0
        condition = lambda num_peaks: False
    elif num_peaks < 2:
        dl = -0.01
        condition = lambda num_peaks: num_peaks < 2
    else:
        dl = 0.01
        condition = lambda num_peaks: num_peaks > 2

    while condition(num_peaks) or level <= 0.0 or level >= 1.0:
        level += dl
        peaks = ui.peaks(prominence_level=level)
        num_peaks = len(peaks)

    return peaks


def find_peaks_with_height(ui: apf.Beat, height=0.3, prominence=0.1) -> np.ndarray:
    from scipy.signal import find_peaks

    peaks, opts = find_peaks(normalize(ui.y), height=height, prominence=prominence)
    if len(peaks) == 1:
        p = (peaks[0], peaks[0])
    elif len(peaks) < 0:
        p = (0, 0)
    else:
        p = (peaks[0], peaks[1])
    return p


def analysis_from_arrays(
    u: np.ndarray,
    v: np.ndarray,
    t: np.ndarray,
    pacing: Optional[np.ndarray] = None,
    background_correction_method="subtract",
    zero_index: Optional[int] = None,
    ignore_pacing: bool = False,
    background_correction_kernel: int = 0,
    threshold_factor: float = 0.5,
    intervals: Optional[List[apf.chopping.Interval]] = None,
) -> Analysis:
    u = apf.utils.numpyfy(u)
    v = apf.utils.numpyfy(v)
    t = apf.utils.numpyfy(t)

    u_beats, intervals = chop_trace(
        u,
        t,
        pacing=pacing,
        intervals=intervals,
        zero_index=zero_index,
        background_correction_kernel=background_correction_kernel,
        threshold_factor=threshold_factor,
        background_correction_method=background_correction_method,
        ignore_pacing=ignore_pacing,
    )

    if len(u_beats) > 1:
        freqs = apf.features.beating_frequency_from_peaks(
            signals=[ui.y for ui in u_beats],
            times=[ui.t for ui in u_beats],
        )
    else:
        freqs = apf.features.beating_frequency_from_apd_line(y=u, time=t)

    freqs = [f for f in freqs if not np.isinf(f) or np.isnan(f)]

    u_peaks = [ui.y.max() for ui in u_beats]
    global_peak_inds = [ui.y.argmax() for ui in u_beats]
    all_peaks_inds = [find_peaks_with_height(ui, 0.3) for ui in u_beats]
    first_peak_inds = []
    for i, peaks in enumerate(all_peaks_inds):
        if len(peaks) == 0:
            peak = global_peak_inds[i]
        else:
            peak = peaks[0]
        first_peak_inds.append(peak)

    u_peaks_first = [ui.y[index] for index, ui in zip(first_peak_inds, u_beats)]
    u_beats_first_normalized = [ui.copy(y_max=p) for ui, p in zip(u_beats, u_peaks_first)]
    time_above_half_height = [ui.time_above_apd_line(0.5) for ui in u_beats]
    time_above_half_height_first = [ui.time_above_apd_line(0.5) for ui in u_beats_first_normalized]
    apd_points_global = [ui.apd_point(50, strategy="big_diff_plus_one") for ui in u_beats]
    apd_points_first = [ui.apd_point(50, strategy="big_diff_plus_one") for ui in u_beats_first_normalized]
    apd_points_global_first_last = [ui.apd_point(50, strategy="first_last") for ui in u_beats]
    apd_points_first_first_last = [ui.apd_point(50, strategy="first_last") for ui in u_beats_first_normalized]

    u_width50_global = [p[1] - p[0] for p in apd_points_global]
    u_width50_first = [p[1] - p[0] for p in apd_points_first]
    u_width50_global_first_last = [p[1] - p[0] for p in apd_points_global_first_last]
    u_width50_first_first_last = [p[1] - p[0] for p in apd_points_first_first_last]
    p = pacing
    if pacing is not None:
        p = pacing[: len(v)]

    v_beats, intervals = chop_trace(
        v,
        t[: len(v)],
        pacing=p,
        intervals=intervals,
        zero_index=zero_index,
        background_correction_kernel=background_correction_kernel,
        threshold_factor=threshold_factor,
        background_correction_method=background_correction_method,
        ignore_pacing=ignore_pacing,
    )
    # time_to_max_contraction_velocty = []
    max_contraction_velocity = []
    # time_to_max_relaxation_velocty = []
    max_relaxation_velocity = []
    time_between_contraction_and_relaxation = []

    for vi in v_beats:
        try:
            t0, t1 = find_two_most_prominent_peaks(vi.y)
        except ProminenceError:
            time_between_contraction_and_relaxation.append(0.0)
            # time_to_max_contraction_velocty.append(vi.t[t0])
            max_contraction_velocity.append(0.0)
            # time_to_max_relaxation_velocty.append(vi.t[t1])
            max_relaxation_velocity.append(0.0)

        else:
            time_between_contraction_and_relaxation.append(vi.t[t1] - vi.t[t0])
            # time_to_max_contraction_velocty.append(vi.t[t0])
            max_contraction_velocity.append(vi.y[t0])
            # time_to_max_relaxation_velocty.append(vi.t[t1])
            max_relaxation_velocity.append(vi.y[t1])
    return Analysis(
        u_peaks=u_peaks,
        u_width50_global=u_width50_global,
        u_peaks_first=u_peaks_first,
        u_width50_first=u_width50_first,
        u_width50_global_first_last=u_width50_global_first_last,
        u_width50_first_first_last=u_width50_first_first_last,
        time_above_half_height=time_above_half_height,
        time_above_half_height_first=time_above_half_height_first,
        max_contraction_velocity=max_contraction_velocity,
        max_relaxation_velocity=max_relaxation_velocity,
        time_between_contraction_and_relaxation=time_between_contraction_and_relaxation,
        frequency=freqs,
    )


def find_two_peaks(y, prominence=1.0):
    peaks, peak_ops = find_peaks(y, prominence=prominence)

    if len(peaks) == 2:
        return peaks, peak_ops, prominence
    elif len(peaks) < 2:
        # We need to lower the prominence
        while len(peaks) < 2 and prominence > 0:
            prominence -= 0.1
            peaks, peak_ops = find_peaks(y, prominence=prominence)
    else:
        # We need to increase the prominence
        while len(peaks) > 2 and prominence < 10.0:
            prominence += 0.1
            peaks, peak_ops = find_peaks(y, prominence=prominence)

    return peaks, peak_ops, prominence


def compute_features(u, v, t, background_correction_method="subtract"):
    u = apf.Beats(
        u,
        t,
        background_correction_method=background_correction_method,
        force_positive=True,
    )
    u_beats = u.beats
    v = apf.Beats(
        v,
        t[: len(v)],
        intervals=u.chopped_data.intervals,
        background_correction_method=background_correction_method,
    )

    v_beats = v.beats

    data = {
        "Maximum rise velocity": [],
        "Peak twitch amplitude": [],
        "Maximum relaxation velocity": [],
        "Beat duration": [],
        "Time to peak twitch amplitude": [],
        "Time to peak contraction velocity": [],
        "Time to peak relaxation velocity": [],
        "Width at half height": [],
    }

    for ui, vi in zip(u_beats, v_beats):
        peaks, peak_ops, prom = find_two_peaks(vi.y)
        try:
            data["Maximum rise velocity"].append(vi.y[peaks[0]])
            data["Peak twitch amplitude"].append(np.max(ui.y))
            data["Maximum relaxation velocity"].append(vi.y[peaks[1]])
            data["Beat duration"].append(ui.t[-1] - ui.t[0])
            data["Time to peak twitch amplitude"].append(ui.ttp())
            data["Time to peak contraction velocity"].append(vi.t[peaks[0]] - vi.t[0])
            data["Time to peak relaxation velocity"].append(vi.t[peaks[1]] - vi.t[0])
            data["Width at half height"].append(ui.apd(50))
        except IndexError:
            continue

    return data
