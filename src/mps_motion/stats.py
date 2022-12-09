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
    u_width50: List[float]
    time_between_contraction_and_relaxation: List[float]
    max_contraction_velocity: List[float]
    max_relaxation_velocity: List[float]

    @property
    def mean(self) -> Dict[str, float]:
        return {
            "u_peaks": np.mean(self.u_peaks),
            "u_width50": np.mean(self.u_width50),
            "time_between_contraction_and_relaxation": np.mean(
                self.time_between_contraction_and_relaxation,
            ),
            "max_contraction_velocity": np.mean(self.max_contraction_velocity),
            "max_relaxation_velocity": np.mean(self.max_relaxation_velocity),
        }

    @property
    def std(self) -> Dict[str, float]:
        return {
            "u_peaks": np.std(self.u_peaks),
            "u_width50": np.std(self.u_width50),
            "time_between_contraction_and_relaxation": np.std(
                self.time_between_contraction_and_relaxation,
            ),
            "max_contraction_velocity": np.std(self.max_contraction_velocity),
            "max_relaxation_velocity": np.std(self.max_relaxation_velocity),
        }


class ProminenceError(RuntimeError):
    pass


def normalize(y: np.ndarray) -> np.ndarray:
    y = np.array(y)
    return (y - y.min()) / (y.max() - y.min())


def width_at_height(y: np.ndarray, t: np.ndarray, height=0.5) -> float:
    pass
    # assert 0 <= height <= 1
    # f = UnivariateSpline(t, normalize(y) - height, s=0, k=3)
    # try:
    #     return np.diff(f.roots())[0]
    # except IndexError:
    #     # Function does not have two zeros
    #     return np.nan


def chop_trace(
    u: np.ndarray,
    t: np.ndarray,
    pacing: Optional[np.ndarray] = None,
    background_correction_method="subtract",
    zero_index: Optional[int] = None,
    ignore_pacing: bool = False,
    intervals: Optional[List[apf.chopping.Interval]] = None,
) -> Tuple[List[apf.Beat], Optional[List[apf.chopping.Interval]]]:
    trace = apf.Beats(
        u,
        t,
        pacing,
        background_correction_method=background_correction_method,
        zero_index=zero_index,
        chopping_options={"ignore_pacing": ignore_pacing},
        intervals=intervals,
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


def find_two_most_prominent_peaks(y) -> Tuple[int, int]:
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
    while len(peaks) < 2 and prominence > 0:
        peaks, opts = find_peaks(normalize(y), prominence=prominence)
        prominence -= 0.1

    if len(peaks) < 2:
        raise ProminenceError("Unable to find two most prominent beats")

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


def analysis_from_arrays(
    u: np.ndarray,
    v: np.ndarray,
    t: np.ndarray,
    pacing: Optional[np.ndarray] = None,
    background_correction_method="subtract",
    zero_index: Optional[int] = None,
    ignore_pacing: bool = False,
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
        background_correction_method=background_correction_method,
        ignore_pacing=ignore_pacing,
    )

    u_peaks = [ui.y.max() for ui in u_beats]
    # u_ttp = [ui.ttp() for ui in u_beats]
    u_width50 = [ui.apd(50) for ui in u_beats]
    p = pacing
    if pacing is not None:
        p = pacing[: len(v)]

    v_beats, intervals = chop_trace(
        v,
        t[: len(v)],
        pacing=p,
        intervals=intervals,
        zero_index=zero_index,
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
        u_width50=u_width50,
        max_contraction_velocity=max_contraction_velocity,
        max_relaxation_velocity=max_relaxation_velocity,
        time_between_contraction_and_relaxation=time_between_contraction_and_relaxation,
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


def compute_features(u, v, t):
    u = apf.Beats(u, t, background_correction_method="subtract")
    u_beats = u.beats
    v = apf.Beats(
        v,
        t[: len(v)],
        intervals=u.chopped_data.intervals,
        background_correction_method="subtract",
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
