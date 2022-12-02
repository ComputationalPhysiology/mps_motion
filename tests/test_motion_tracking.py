from unittest import mock

import dask.array as da
import numpy as np
import pytest
from mps_motion import FLOW_ALGORITHMS as _FLOW_ALGORITHMS
from mps_motion import Mechanics
from mps_motion import motion_tracking
from mps_motion import OpticalFlow
from mps_motion import utils

# from typing import Type
FLOW_ALGORITHMS = [alg for alg in dir(_FLOW_ALGORITHMS) if not alg.startswith("_")]


@pytest.mark.parametrize("reference_frame", ["min", "max", "median", "mean"])
def test_reference_frame_np(
    test_data: utils.MPSData,
    reference_frame: str,
):
    reference_image = motion_tracking.get_reference_image(
        reference_frame=reference_frame,
        frames=test_data.frames,
    )
    ref = getattr(np, reference_frame)(test_data.frames, axis=2)
    assert np.all(ref == reference_image)


@pytest.mark.parametrize(
    "reference_index, interval",
    [(0, (0, 3)), (1, (0, 3)), (2, (1, 4)), (-1, (-3, -1))],
)
def test_reference_frame_digit(
    test_data: utils.MPSData,
    reference_index: int,
    interval,
):

    reference_frame = test_data.time_stamps[int(reference_index)]
    if reference_index == -1:
        interval = (interval[0], test_data.time_stamps.size)

    ref = test_data.frames[:, :, interval[0] : interval[1]].mean(-1)
    reference_image = motion_tracking.get_reference_image(
        reference_frame=reference_frame,
        frames=test_data.frames,
        time_stamps=test_data.time_stamps,
    )
    assert np.all(abs(ref - reference_image) < 1e-12)


@pytest.mark.parametrize(
    "reference_index, interval",
    [("0", (0, 3)), ("1", (0, 3)), ("2", (1, 4)), ("-1", (-3, -1))],
)
def test_reference_frame_digit_str(
    test_data: utils.MPSData,
    reference_index: str,
    interval,
):
    reference_frame = test_data.time_stamps[int(reference_index)]

    if reference_index == "-1":
        interval = (interval[0], test_data.time_stamps.size)

    reference_image = motion_tracking.get_reference_image(
        reference_frame=reference_frame,
        frames=test_data.frames,
        time_stamps=test_data.time_stamps,
    )
    ref = test_data.frames[:, :, interval[0] : interval[1]].mean(-1)
    assert np.all(abs(ref - reference_image) < 1e-12)


@pytest.mark.parametrize("reference_frame", ["a", "std", ""])
def test_reference_frame_invalid(
    test_data: utils.MPSData,
    reference_frame: str,
):
    with pytest.raises(ValueError):
        motion_tracking.get_reference_image(
            reference_frame=reference_frame,
            frames=test_data.frames,
        )


@pytest.mark.parametrize("flow_algorithm", ["", "dslkgm"])
def test_invalid_algoritm(
    test_data: utils.MPSData,
    flow_algorithm: _FLOW_ALGORITHMS,
):
    with pytest.raises(ValueError):
        OpticalFlow(test_data, flow_algorithm=flow_algorithm)


@pytest.mark.parametrize("flow_algorithm", FLOW_ALGORITHMS)
def test_get_displacement_lazy(
    test_data: utils.MPSData,
    flow_algorithm: _FLOW_ALGORITHMS,
):
    np.random.seed(1)
    arr = np.random.random((4, 5, 3, 2))  # (width, heighth, num_time_points)
    with mock.patch(f"mps_motion.{flow_algorithm}.get_displacements") as _mock:
        _mock.return_value = arr
        m = OpticalFlow(test_data, flow_algorithm=flow_algorithm)
        U = m.get_displacements()
        assert (U[:, :, 0, :] == arr[:, :, 0, :]).all().compute()

    _mock.assert_called_once()


@pytest.mark.parametrize("flow_algorithm", FLOW_ALGORITHMS)
def test_get_displacement(test_data: utils.MPSData, flow_algorithm: _FLOW_ALGORITHMS):
    m = OpticalFlow(test_data, flow_algorithm=flow_algorithm)
    disp = m.get_displacements()
    assert disp.shape == (test_data.size_x, test_data.size_y, test_data.num_frames, 2)


def test_get_displacement_unit(test_data: utils.MPSData):

    m = OpticalFlow(test_data)
    disp_px = m.get_displacements(unit="pixels")
    disp_um = m.get_displacements(unit="um", recompute=True)

    assert da.isclose(
        disp_um.x.mean().max(),
        disp_px.x.mean().max() * test_data.info["um_per_pixel"],
    )

    assert da.isclose(
        disp_um.max().max(),
        disp_px.max().max() * test_data.info["um_per_pixel"],
    )
    assert da.isclose(
        disp_um.norm().max().max(),
        disp_px.norm().max().max() * test_data.info["um_per_pixel"],
    )


@pytest.mark.parametrize(
    "unit",
    ["um", "pixels"],
)
def test_get_displacement_scale(
    test_data: utils.MPSData,
    unit: str,
):
    """Test that there are now exceptions raised"""
    m = OpticalFlow(test_data)
    u_full = m.get_displacements(unit=unit)
    u = m.get_displacements(unit=unit, scale=0.5)
    assert np.isclose(u.mean().max().compute(), u_full.mean().max().compute())


@pytest.mark.parametrize(
    "unit",
    ["um", "pixels"],
)
def test_get_velocity_scale(
    test_data: utils.MPSData,
    unit: str,
):
    """Test that there are now exceptions raised"""
    m = OpticalFlow(test_data)
    v_full = m.get_velocities(unit=unit)
    v = m.get_velocities(unit=unit, scale=0.5)
    u_full = m.get_displacements(unit=unit, scale=0.5)
    v_full_from_u = Mechanics(u_full, t=test_data.time_stamps).velocity()

    u = m.get_displacements(unit=unit)
    v_from_u = Mechanics(u, t=test_data.time_stamps).velocity()

    # We cannot expected these to be all equal, but they should be
    # of the save order, so a 5-10% difference is OK
    assert np.isclose(v.mean().max().compute(), v_full.mean().max().compute(), rtol=0.1)

    assert np.isclose(
        v_from_u.mean().max().compute(),
        v_full_from_u.mean().max().compute(),
        rtol=0.1,
    )
    assert np.isclose(
        v_from_u.mean().max().compute(),
        v.mean().max().compute(),
        rtol=0.1,
    )


def test_OpticalFlow_options(test_data: utils.MPSData):

    step = 4
    m = OpticalFlow(test_data, flow_algorithm=_FLOW_ALGORITHMS.lucas_kanade, step=step)
    assert m.options["step"] == step


def test_estimate_reference_index(synthetic_trace):

    t = synthetic_trace.t[:-1]
    v = v = synthetic_trace.v

    index = motion_tracking.estimate_referece_image_from_velocity(
        t=t,
        v=v,
        rel_tol=0.001,
    )
    assert index == 87
