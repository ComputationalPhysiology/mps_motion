import dask.array as da
import numpy as np
import pytest

from mps_motion_tracking import Mechancis


def test_E_symmetry(mech_obj):

    E = mech_obj.E
    e12 = E[:, :, :, 0, 1]
    e21 = E[:, :, :, 1, 0]
    assert da.allclose(e12, e21).compute()


def test_principal_strain(mech_obj):
    e1 = mech_obj.principal_strain[:, :, :, 0]

    assert np.allclose(e1[:, :, 0], 0)

    a1 = np.array([[0.005, 0.05], [0.05, 0.0]])
    eig = np.linalg.eigvals(a1)
    assert np.allclose(e1[:, :, 1], eig.min())

    e2 = mech_obj.principal_strain[:, :, :, 1]

    assert np.allclose(e2[:, :, 1], eig.max())


@pytest.fixture
def mech_obj():

    width = 10
    height = 15
    num_time_points = 4

    u = np.zeros((height, width, 2, num_time_points))
    # First time point is zero
    # Second time point has a linar displacement in x
    u[:, :, 0, 1] = np.fromfunction(
        lambda y, x: x / width, shape=(height, width), dtype=float
    )
    # Third points have linear displacement in y
    u[:, :, 1, 2] = np.fromfunction(
        lambda y, x: y / height, shape=(height, width), dtype=float
    )
    # Forth is linear in both
    u[:, :, 0, 3] = u[:, :, 0, 1]
    u[:, :, 1, 3] = u[:, :, 1, 2]

    return Mechancis(u)


def test_shapes():

    width = 10
    height = 12
    num_time_steps = 14

    u = np.random.random((width, height, 2, num_time_steps))
    m = Mechancis(u)

    assert m.u.shape == (width, height, num_time_steps, 2)
    assert m.du.shape == (width, height, num_time_steps, 2, 2)
    assert m.F.shape == (width, height, num_time_steps, 2, 2)
    assert m.E.shape == (width, height, num_time_steps, 2, 2)
    assert m.principal_strain.shape == (width, height, num_time_steps, 2)


if __name__ == "__main__":
    # main()
    # mech_obj()
    test_shapes()
