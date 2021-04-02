import itertools as it
from unittest import mock

import numpy as np
import pytest

from mps_motion_tracking import FLOW_ALGORITHMS as _FLOW_ALGORITHMS
from mps_motion_tracking import OpticalFlow, utils

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


@pytest.mark.parametrize("reference_index", [0, 1, -1])
def test_reference_frame_digit(
    test_data: utils.MPSData,
    reference_index: int,
):
    reference_frame = test_data.time_stamps[int(reference_index)]
    m = OpticalFlow(test_data, reference_frame=reference_frame)

    assert abs(float(m.reference_frame) - reference_frame) < 1e-8
    ref = test_data.frames[:, :, int(reference_index)]
    assert np.all(ref == m.reference_image)


@pytest.mark.parametrize("reference_index", ["0", "1", "-1"])
def test_reference_frame_digit_str(
    test_data: utils.MPSData,
    reference_index: str,
):
    reference_frame = test_data.time_stamps[int(reference_index)]
    m = OpticalFlow(test_data, reference_frame=reference_frame)
    assert abs(float(m.reference_frame) - reference_frame) < 1e-8
    ref = test_data.frames[:, :, int(reference_index)]
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
        m.get_displacements(raw=True)
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
            disp_um.x.mean().max().compute()
            - disp_px.x.mean().max().compute() * test_data.info["um_per_pixel"]
        )
        < 1e-12
    )


@pytest.mark.parametrize(
    "flow_algorithm, unit", it.product(FLOW_ALGORITHMS, ["um", "pixels"])
)
def test_get_displacement_scale_algs(
    test_data: utils.MPSData, flow_algorithm: str, unit: str
):
    """Test that there are now exceptions raised"""
    m = OpticalFlow(test_data, flow_algorithm=flow_algorithm)
    m.get_displacements(unit=unit, scale=0.5)
