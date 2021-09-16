import itertools as it
from unittest import mock

import numpy as np
import pytest

from mps_motion_tracking import FLOW_ALGORITHMS as _FLOW_ALGORITHMS
from mps_motion_tracking import OpticalFlow
from mps_motion_tracking import utils

# from typing import Type
FLOW_ALGORITHMS = [alg for alg in dir(_FLOW_ALGORITHMS) if not alg.startswith("_")]


@pytest.mark.parametrize("reference_frame", ["min", "max", "median", "mean"])
def test_reference_frame_np(
    test_data: utils.MPSData,
    reference_frame: str,
):
    m = OpticalFlow(test_data, reference_frame=reference_frame)
    assert m.reference_frame == reference_frame
    ref = getattr(np, reference_frame)(test_data.frames, axis=2)
    assert np.all(ref == m.reference_image)


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
    m = OpticalFlow(test_data, reference_frame=reference_frame)
    if reference_index == -1:
        interval = (interval[0], test_data.time_stamps.size)

    assert abs(float(m.reference_frame) - reference_frame) < 1e-8
    ref = test_data.frames[:, :, interval[0] : interval[1]].mean(-1)

    assert np.all(abs(ref - m.reference_image) < 1e-12)


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
    m = OpticalFlow(test_data, reference_frame=reference_frame)
    if reference_index == "-1":
        interval = (interval[0], test_data.time_stamps.size)

    assert abs(float(m.reference_frame) - reference_frame) < 1e-8
    ref = test_data.frames[:, :, interval[0] : interval[1]].mean(-1)
    assert np.all(abs(ref - m.reference_image) < 1e-12)


@pytest.mark.parametrize("reference_frame", ["a", "std", ""])
def test_reference_frame_invalid(
    test_data: utils.MPSData,
    reference_frame: str,
):
    with pytest.raises(ValueError):
        OpticalFlow(test_data, reference_frame=reference_frame)


@pytest.mark.parametrize("flow_algorithm", ["", "dslkgm"])
def test_invalid_algoritm(
    test_data: utils.MPSData,
    flow_algorithm: str,
):
    with pytest.raises(ValueError):
        OpticalFlow(test_data, flow_algorithm=flow_algorithm)


@pytest.mark.parametrize("flow_algorithm", FLOW_ALGORITHMS)
def test_get_displacement_lazy(
    test_data: utils.MPSData,
    flow_algorithm: str,
):
    np.random.seed(1)
    arr = np.random.random((4, 5, 2, 3))  # (width, heighth, 2, num_time_points)
    with mock.patch(f"mps_motion_tracking.{flow_algorithm}.get_displacements") as _mock:
        _mock.return_value = arr
        m = OpticalFlow(test_data, flow_algorithm=flow_algorithm)
        U = m.get_displacements()
        assert (U[:, :, 0, :] == arr[:, :, :, 0]).all().compute()

    _mock.assert_called_once()


@pytest.mark.parametrize("flow_algorithm", FLOW_ALGORITHMS)
def test_get_displacement(test_data: utils.MPSData, flow_algorithm: str):
    m = OpticalFlow(test_data, flow_algorithm=flow_algorithm)
    disp = m.get_displacements()
    assert disp.shape == (test_data.size_x, test_data.size_y, test_data.num_frames, 2)


def test_get_displacement_unit(test_data: utils.MPSData):

    m = OpticalFlow(test_data)
    disp_px = m.get_displacements(unit="pixels")
    disp_um = m.get_displacements(unit="um", recompute=True)

    assert (
        abs(
            disp_um.x.mean().max().compute()  # type: ignore
            - disp_px.x.mean().max().compute() * test_data.info["um_per_pixel"],  # type: ignore
        )
        < 1e-12
    )

    assert (
        abs(
            disp_um.max().max().compute()  # type: ignore
            - disp_px.max().max().compute() * test_data.info["um_per_pixel"],  # type: ignore
        )
        < 1e-12
    )

    assert (
        abs(
            disp_um.norm().max().max().compute()  # type: ignore
            - disp_px.norm().max().max().compute() * test_data.info["um_per_pixel"],  # type: ignore
        )
        < 1e-12
    )


@pytest.mark.parametrize(
    "flow_algorithm, unit",
    it.product(FLOW_ALGORITHMS, ["um", "pixels"]),
)
def test_get_displacement_scale_algs(
    test_data: utils.MPSData,
    flow_algorithm: str,
    unit: str,
):
    """Test that there are now exceptions raised"""
    m = OpticalFlow(test_data, flow_algorithm=flow_algorithm)
    m.get_displacements(unit=unit, scale=0.5)


def test_OpticalFlow_options(test_data: utils.MPSData):

    step = 4
    m = OpticalFlow(test_data, flow_algorithm="lucas_kanade", step=step)
    assert m.options["step"] == step
