import itertools as it
from typing import Type
from unittest import mock

import numpy as np
import pytest

from mps_motion_tracking import motion_tracking, utils

OptFlows = [motion_tracking.DenseOpticalFlow, motion_tracking.SparseOpticalFlow]


@pytest.mark.parametrize(
    "reference_frame, OptFlow", it.product(["min", "max", "median", "mean"], OptFlows)
)
def test_reference_frame_np(
    test_data: utils.MPSData,
    reference_frame: str,
    OptFlow: Type[motion_tracking.OpticalFlow],
):
    m = OptFlow(test_data, reference_frame=reference_frame)
    assert m.reference_frame == reference_frame
    ref = getattr(np, reference_frame)(test_data.frames, axis=2)
    assert np.all(ref == m.reference_image)


@pytest.mark.parametrize("reference_frame, OptFlow", it.product([0, 1, -1], OptFlows))
def test_reference_frame_digit(
    test_data: utils.MPSData,
    reference_frame: int,
    OptFlow: Type[motion_tracking.OpticalFlow],
):
    m = OptFlow(test_data, reference_frame=reference_frame)
    assert m.reference_frame == str(reference_frame)
    ref = test_data.frames[:, :, reference_frame]
    assert np.all(ref == m.reference_image)


@pytest.mark.parametrize(
    "reference_frame, OptFlow", it.product(["0", "1", "-1"], OptFlows)
)
def test_reference_frame_digit_str(
    test_data: utils.MPSData,
    reference_frame: str,
    OptFlow: Type[motion_tracking.OpticalFlow],
):
    m = OptFlow(test_data, reference_frame=reference_frame)
    assert m.reference_frame == reference_frame
    ref = test_data.frames[:, :, int(reference_frame)]
    assert np.all(ref == m.reference_image)


@pytest.mark.parametrize(
    "reference_frame, OptFlow", it.product(["a", "std", ""], OptFlows)
)
def test_reference_frame_invalid(
    test_data: utils.MPSData,
    reference_frame: str,
    OptFlow: Type[motion_tracking.OpticalFlow],
):
    with pytest.raises(ValueError):
        OptFlow(test_data, reference_frame=reference_frame)


@pytest.mark.parametrize(
    "flow_algorithm, OptFlow", it.product(["", "dslkgm"], OptFlows)
)
def test_invalid_algoritm(
    test_data: utils.MPSData,
    flow_algorithm: str,
    OptFlow: Type[motion_tracking.OpticalFlow],
):
    with pytest.raises(ValueError):
        OptFlow(test_data, flow_algorithm=flow_algorithm)


@pytest.mark.parametrize(
    "flow_algorithm, OptFlow",
    it.chain(
        zip(
            motion_tracking.DENSE_FLOW_ALGORITHMS,
            it.repeat(motion_tracking.DenseOpticalFlow),
        ),
        zip(
            motion_tracking.SPARSE_FLOW_ALGORITHMS,
            it.repeat(motion_tracking.SparseOpticalFlow),
        ),
    ),
)
def test_get_displacement_lazy(
    test_data: utils.MPSData,
    flow_algorithm: str,
    OptFlow: Type[motion_tracking.OpticalFlow],
):

    with mock.patch(f"mps_motion_tracking.{flow_algorithm}.get_displacements") as _mock:
        m = OptFlow(test_data, flow_algorithm=flow_algorithm)
        m.get_displacements()
    _mock.assert_called_once()


@pytest.mark.parametrize("flow_algorithm", motion_tracking.DENSE_FLOW_ALGORITHMS)
def test_get_displacement_dense(test_data: utils.MPSData, flow_algorithm: str):
    m = motion_tracking.DenseOpticalFlow(test_data, flow_algorithm=flow_algorithm)

    disp = m.get_displacements()
    assert disp.shape == (test_data.size_x, test_data.size_y, 2, test_data.num_frames)
