import numpy as np
from mps_motion import stats


def test_analysis_from_mechanics(mech_trace_obj):

    analysis = stats.analysis_from_mechanics(mech_trace_obj)

    mean_values = analysis.mean
    std = analysis.std
    assert np.isclose(mean_values["max_contraction_velocity"], 5125.387931222401)
    assert np.isclose(mean_values["max_relaxation_velocity"], 4640.270308088497)
    assert np.isclose(
        mean_values["time_between_contraction_and_relaxation"],
        266.8879394531132,
    )
    assert np.isclose(mean_values["u_peaks"], 0.4589306341013011)
    assert np.isclose(mean_values["u_width50"], 260.4451501720163)

    assert np.isclose(std["max_contraction_velocity"], 161.86832160836673)
    assert np.isclose(std["max_relaxation_velocity"], 178.25122955064532)
    assert np.isclose(std["time_between_contraction_and_relaxation"], 4.707889720282436)
    assert np.isclose(std["u_peaks"], 0.0038838100228260463)
    assert np.isclose(std["u_width50"], 1.1983554291640317)


def test_analysis_from_arrays(mech_trace_obj):

    u = mech_trace_obj.u.norm().mean()
    t = mech_trace_obj.t
    v = mech_trace_obj.velocity().norm().mean()

    analysis = stats.analysis_from_arrays(u, v, t)

    mean_values = analysis.mean
    std = analysis.std
    assert np.isclose(mean_values["max_contraction_velocity"], 5125.387931222401)
    assert np.isclose(mean_values["max_relaxation_velocity"], 4640.270308088497)
    assert np.isclose(
        mean_values["time_between_contraction_and_relaxation"],
        266.8879394531132,
    )
    assert np.isclose(mean_values["u_peaks"], 0.4589306341013011)
    assert np.isclose(mean_values["u_width50"], 260.4451501720163)

    assert np.isclose(std["max_contraction_velocity"], 161.86832160836673)
    assert np.isclose(std["max_relaxation_velocity"], 178.25122955064532)
    assert np.isclose(std["time_between_contraction_and_relaxation"], 4.707889720282436)
    assert np.isclose(std["u_peaks"], 0.0038838100228260463)
    assert np.isclose(std["u_width50"], 1.1983554291640317)
