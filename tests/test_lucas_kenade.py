import numpy as np

from mps_motion_tracking import lucas_kanade as lk


def test_flow():

    reference_image = 255 * np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
    image = 255 * np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)

    step = 16
    h, w = reference_image.shape[:2]
    grid = np.mgrid[step / 2 : w : step, step / 2 : h : step].astype(int)
    reference_points = np.expand_dims(grid.astype(np.float32).reshape(2, -1).T, 1)
    flow = lk.flow(image, reference_image, reference_points)
    assert flow.shape == (reference_points.shape[0], reference_points.shape[2])
    assert flow.dtype == np.float32


def test_flow_map():

    reference_image = 255 * np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
    image = 255 * np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)

    step = 16
    h, w = reference_image.shape[:2]
    grid = np.mgrid[step / 2 : w : step, step / 2 : h : step].astype(int)
    reference_points = np.expand_dims(grid.astype(np.float32).reshape(2, -1).T, 1)
    flow = lk.flow_map((image, reference_image, reference_points))
    assert flow.shape == (reference_points.shape[0], reference_points.shape[2])
    assert flow.dtype == np.float32


if __name__ == "__main__":
    test_flow()
    test_flow_map()
