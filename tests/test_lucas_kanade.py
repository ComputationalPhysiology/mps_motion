import numpy as np
import pytest

from mps_motion_tracking import lucas_kanade as lk


@pytest.mark.parametrize(
    "size, step, interpolate, resize, expected_shape, expected_type",
    [
        ((64, 64), 16, False, False, (16, 2), np.float32),
        ((64, 64), 16, False, True, (64, 64, 2), np.float64),
        ((64, 64), 16, True, False, (64, 64, 2), np.float64),
        ((64, 64), 16, True, True, (64, 64, 2), np.float64),
    ],
)
def test_flow_shape(size, step, interpolate, resize, expected_shape, expected_type):

    reference_image = 255 * np.random.randint(0, 255, size=size, dtype=np.uint8)
    image = 255 * np.random.randint(0, 255, size=size, dtype=np.uint8)

    reference_points = lk.get_uniform_reference_points(reference_image, step)
    flow = lk.flow(
        image, reference_image, reference_points, interpolate=interpolate, resize=resize
    )

    assert flow.shape == expected_shape
    assert flow.dtype == expected_type


@pytest.mark.parametrize(
    "input_shape, step",
    [
        ((64, 64), 48),
        ((128, 128), 48),
        ((300, 200), 48),
        ((100, 50), 12),
        ((121, 241), 13),
    ],
)
def test_get_uniform_reference_points(input_shape, step):
    ref_points = lk.get_uniform_reference_points(np.zeros(input_shape), step=step)
    assert ref_points.shape == (
        ((input_shape[0] + step // 2) // step) * ((input_shape[1] + step // 2) // step),
        1,
        2,
    )


def test_get_displacements():

    size = (64, 64)
    reference_image = 255 * np.random.randint(0, 255, size=size, dtype=np.uint8)
    image = 255 * np.random.randint(0, 255, size=size, dtype=np.uint8)
    frames = [reference_image, image, image]
    u = lk.get_displacements(
        frames=np.array(frames).T,
        reference_image=reference_image,
    )
    assert u.shape == (size[0], size[1], 2, len(frames))


def test_flow_map():

    reference_image = 255 * np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
    image = 255 * np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)

    step = 16
    reference_points = lk.get_uniform_reference_points(reference_image, step)
    flow = lk.flow_map((image, reference_image, reference_points))
    assert flow.shape == (reference_points.shape[0], reference_points.shape[2])
    assert flow.dtype == np.float32
