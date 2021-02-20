import itertools as it
from unittest import mock

import numpy as np
import pytest

from mps_motion_tracking import FLOW_ALGORITHMS, OpticalFlow, utils

# from typing import Type


@pytest.mark.parametrize("reference_frame", ["min", "max", "median", "mean"])
def test_reference_frame_np(
    test_data: utils.MPSData,
    reference_frame: str,
):
    m = OpticalFlow(test_data, reference_frame=reference_frame)
    assert m.reference_frame == reference_frame
    ref = getattr(np, reference_frame)(test_data.frames, axis=2)
    assert np.all(ref == m.reference_image)


@pytest.mark.parametrize("reference_frame", [0, 1, -1])
def test_reference_frame_digit(
    test_data: utils.MPSData,
    reference_frame: int,
):
    m = OpticalFlow(test_data, reference_frame=reference_frame)
    assert m.reference_frame == str(reference_frame)
    ref = test_data.frames[:, :, reference_frame]
    assert np.all(ref == m.reference_image)


@pytest.mark.parametrize("reference_frame", ["0", "1", "-1"])
def test_reference_frame_digit_str(
    test_data: utils.MPSData,
    reference_frame: str,
):
    m = OpticalFlow(test_data, reference_frame=reference_frame)
    assert m.reference_frame == reference_frame
    ref = test_data.frames[:, :, int(reference_frame)]
    assert np.all(ref == m.reference_image)


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

    with mock.patch(f"mps_motion_tracking.{flow_algorithm}.get_displacements") as _mock:
        m = OpticalFlow(test_data, flow_algorithm=flow_algorithm)
        m.get_displacements()
    _mock.assert_called_once()


@pytest.mark.parametrize("flow_algorithm", FLOW_ALGORITHMS)
def test_get_displacement(test_data: utils.MPSData, flow_algorithm: str):
    m = OpticalFlow(test_data, flow_algorithm=flow_algorithm)
    disp = m.get_displacements()
    assert disp.shape == (test_data.size_x, test_data.size_y, 2, test_data.num_frames)


def test_get_displacement_unit(test_data: utils.MPSData):

    m = OpticalFlow(test_data)
    disp_px = m.get_displacements(unit="pixels")
    disp_um = m.get_displacements(unit="um")

    assert abs(disp_um.max() - disp_px.max() * test_data.info["um_per_pixel"]) < 1e-12


@pytest.mark.parametrize(
    "flow_algorithm, unit", it.product(FLOW_ALGORITHMS, ["um", "pixels"])
)
def test_get_displacement_scale_algs(
    test_data: utils.MPSData, flow_algorithm: str, unit: str
):
    """Test that there are now exceptions raised"""
    m = OpticalFlow(test_data, flow_algorithm=flow_algorithm)
    m.get_displacements(unit=unit, scale=0.5)


@pytest.mark.parametrize("unit", ["um", "pixels"])
def test_get_displacement_scale(test_data: utils.MPSData, unit):

    m = OpticalFlow(test_data)

    disp_1 = m.get_displacements(unit=unit, scale=1.0)
    disp_08 = m.get_displacements(unit=unit, scale=0.8)
    disp_05 = m.get_displacements(unit=unit, scale=0.5)
    disp_03 = m.get_displacements(unit=unit, scale=0.3)
    disp_02 = m.get_displacements(unit=unit, scale=0.2)

    maxs = []
    mins = []
    for d in [disp_1, disp_08, disp_05, disp_03, disp_02]:
        maxs.append(d.max())
        mins.append(d.min())

    assert not all(x < y for x, y in zip(maxs, maxs[1:]))
    assert not all(x > y for x, y in zip(maxs, maxs[1:]))
    assert not all(x < y for x, y in zip(mins, mins[1:]))
    assert not all(x > y for x, y in zip(mins, mins[1:]))
