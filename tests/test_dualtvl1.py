import numpy as np
import pytest
from mps_motion import dualtvl1

docutils = pytest.importorskip("cv2.optflow")


def test_flow():

    reference_image = 255 * np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
    image = 255 * np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)

    flow = dualtvl1.flow(image, reference_image)
    assert flow.shape == (reference_image.shape[0], reference_image.shape[1], 2)
    assert flow.dtype == np.float32
