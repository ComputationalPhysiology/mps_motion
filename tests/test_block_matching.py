import numpy as np
import pytest
from mps_motion import block_matching as bm


@pytest.mark.parametrize(
    "size, block_size, resize, expected_shape, expected_type",
    [
        ((64, 64), 9, False, (7, 7, 2), np.float64),
        ((64, 64), 9, True, (64, 64, 2), np.float64),
    ],
)
def test_flow_shape(size, block_size, resize, expected_shape, expected_type):

    reference_image = 255 * np.random.randint(0, 255, size=size, dtype=np.uint8)
    image = 255 * np.random.randint(0, 255, size=size, dtype=np.uint8)

    flow = bm.flow(
        image=image,
        reference_image=reference_image,
        block_size=block_size,
        resize=resize,
    )
    assert flow.shape == expected_shape
    assert flow.dtype == expected_type


def test_get_displacements():

    size = (64, 64)
    reference_image = 255 * np.random.randint(0, 255, size=size, dtype=np.uint8)
    image = 255 * np.random.randint(0, 255, size=size, dtype=np.uint8)
    frames = [reference_image, image, image]
    u = bm.get_displacements(
        frames=np.array(frames).T,
        reference_image=reference_image,
    )
    assert u.shape == (size[0], size[1], len(frames), 2)
