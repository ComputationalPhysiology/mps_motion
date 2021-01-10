import numpy as np

from mps_motion_tracking import farneback as fb


def test_flow():

    reference_image = 255 * np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
    image = 255 * np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)

    flow = fb.flow(image, reference_image)
    assert flow.shape == (reference_image.shape[0], reference_image.shape[1], 2)
    assert flow.dtype == np.float32


def test_flow_map():

    reference_image = 255 * np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
    image = 255 * np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)

    flow = fb.flow_map((image, reference_image))
    assert flow.shape == (reference_image.shape[0], reference_image.shape[1], 2)
    assert flow.dtype == np.float32


if __name__ == "__main__":
    test_flow()
    test_flow_map()
