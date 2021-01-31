# import itertools as it

import numpy as np
import pytest

from mps_motion_tracking import OpticalFlow, utils

# from typing import Type

# from unittest import mock


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


# @pytest.mark.parametrize(
#     "flow_algorithm",
#     it.chain(
#         zip(
#             motion_tracking.DENSE_FLOW_ALGORITHMS,
#             it.repeat(motion_tracking.DenseOpticalFlow),
#         ),
#         zip(
#             motion_tracking.SPARSE_FLOW_ALGORITHMS,
#             it.repeat(motion_tracking.SparseOpticalFlow),
#         ),
#     ),
# )
# def test_get_displacement_lazy(
#     test_data: utils.MPSData,
#     flow_algorithm: str,
# ):

#     with mock.patch(f"mps_motion_tracking.{flow_algorithm}.get_displacements") as _mock:
#         m = OpticalFlow(test_data, flow_algorithm=flow_algorithm)
#         m.get_displacements()
#     _mock.assert_called_once()


# @pytest.mark.parametrize("flow_algorithm", motion_tracking.DENSE_FLOW_ALGORITHMS)
# def test_get_displacement_dense(test_data: utils.MPSData, flow_algorithm: str):
#     m = motion_tracking.DenseOpticalFlow(test_data, flow_algorithm=flow_algorithm)

#     disp = m.get_displacements()
#     assert disp.shape == (test_data.size_x, test_data.size_y, 2, test_data.num_frames)
