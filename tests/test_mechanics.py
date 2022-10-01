import dask.array as da
import numpy as np
import pytest
from mps_motion import frame_sequence as fs
from mps_motion import Mechanics
from mps_motion import mechanics


def test_deformation_gradient():

    width = 10
    height = 15

    a11 = 1 / width
    a21 = -1 / height
    a12 = -1 / width
    a22 = 1 / height

    u = np.zeros((width, height, 1, 2))
    u[:, :, 0, 0] = np.fromfunction(
        lambda x, y: a11 * x + a21 * y,
        shape=(width, height),
        dtype=float,
    )
    u[:, :, 0, 1] = np.fromfunction(
        lambda x, y: a12 * x + a22 * y,
        shape=(width, height),
        dtype=float,
    )
    dx = 1
    U = fs.VectorFrameSequence(u, dx=dx)
    m = Mechanics(U)

    du = np.array([[a11 / dx, a12 / dx], [a21 / dx, a22 / dx]])
    F = du + np.eye(2)

    assert da.isclose(m.F[:, :, :, 0, 0], F[0, 0]).all().compute()
    assert da.isclose(m.F[:, :, :, 0, 1], F[0, 1]).all().compute()
    assert da.isclose(m.F[:, :, :, 1, 0], F[1, 0]).all().compute()
    assert da.isclose(m.F[:, :, :, 1, 1], F[1, 1]).all().compute()


def test_strain():

    width = 10
    height = 15

    a11 = 1 / width
    a21 = -1 / height
    a12 = -1 / width
    a22 = 1 / height

    u = np.zeros((width, height, 1, 2))
    u[:, :, 0, 0] = np.fromfunction(
        lambda x, y: a11 * x + a21 * y,
        shape=(width, height),
        dtype=float,
    )
    u[:, :, 0, 1] = np.fromfunction(
        lambda x, y: a12 * x + a22 * y,
        shape=(width, height),
        dtype=float,
    )
    dx = 1
    U = fs.VectorFrameSequence(u, dx=dx)
    m = Mechanics(U)

    du = np.array([[a11 / dx, a12 / dx], [a21 / dx, a22 / dx]])
    F = du + np.eye(2)
    E = 0.5 * (F.T.dot(F) - np.eye(2))

    assert da.isclose(m.E[:, :, :, 0, 0], E[0, 0]).all().compute()
    assert da.isclose(m.E[:, :, :, 0, 1], E[0, 1]).all().compute()
    assert da.isclose(m.E[:, :, :, 1, 0], E[1, 0]).all().compute()
    assert da.isclose(m.E[:, :, :, 1, 1], E[1, 1]).all().compute()


def test_principal_strain(mech_obj):
    e1 = mech_obj.principal_strain[:, :, :, 0]

    assert np.allclose(e1[:, :, 0], 0)

    a1 = np.array([[0.005, 0.05], [0.05, 0.0]])
    eig = np.linalg.eigvals(a1)
    assert np.allclose(e1[:, :, 1], eig.min())

    e2 = mech_obj.principal_strain[:, :, :, 1]

    assert np.allclose(e2[:, :, 1], eig.max())


@pytest.mark.parametrize("dx", [1.0, 0.5, 2.0])
def test_dx(dx):
    # f(x, y) = (x / width - y / height & y / height - x / width)
    # Df = (1/width & -1/height // -1/width & 1/height)

    width = 10
    height = 15

    a11 = 1 / width
    a21 = -1 / height
    a12 = -1 / width
    a22 = 1 / height

    u = np.zeros((width, height, 1, 2))
    u[:, :, 0, 0] = np.fromfunction(
        lambda x, y: a11 * x + a21 * y,
        shape=(width, height),
        dtype=float,
    )
    u[:, :, 0, 1] = np.fromfunction(
        lambda x, y: a12 * x + a22 * y,
        shape=(width, height),
        dtype=float,
    )
    dx = 1
    U = fs.VectorFrameSequence(u, dx=dx)
    m = Mechanics(U)

    assert da.isclose(m.du[:, :, :, 0, 0], a11 / dx).all().compute()
    assert da.isclose(m.du[:, :, :, 0, 1], a12 / dx).all().compute()
    assert da.isclose(m.du[:, :, :, 1, 0], a21 / dx).all().compute()
    assert da.isclose(m.du[:, :, :, 1, 1], a22 / dx).all().compute()


def test_velocity(mech_obj):
    v = mech_obj.velocity() * 1000
    assert np.isclose(
        v[:, :, 0, 0],
        mech_obj.u[:, :, 1, 0] - mech_obj.u[:, :, 0, 0],
    ).all()
    assert np.isclose(
        v[:, :, 1, 0],
        mech_obj.u[:, :, 2, 0] - mech_obj.u[:, :, 1, 0],
    ).all()
    assert np.isclose(
        v[:, :, 0, 1],
        mech_obj.u[:, :, 1, 1] - mech_obj.u[:, :, 0, 1],
    ).all()
    assert np.isclose(
        v[:, :, 1, 1],
        mech_obj.u[:, :, 2, 1] - mech_obj.u[:, :, 1, 1],
    ).all()


def test_compute_displacement(mech_obj):
    spacing = 1  # Only implemented for spacings of 1
    v = mech_obj.velocity(spacing=spacing)
    u = mechanics.compute_displacement(
        v.array,
        mech_obj.t,
        ref_index=0,
        spacing=spacing,
    )
    assert da.isclose(u, mech_obj.u.array).all().compute()


def test_shapes():

    width = 10
    height = 12
    num_time_steps = 14

    u = np.random.random((width, height, num_time_steps, 2))
    m = Mechanics(fs.VectorFrameSequence(u))

    assert m.u.shape == (width, height, num_time_steps, 2)
    assert m.du.shape == (width, height, num_time_steps, 2, 2)
    assert m.F.shape == (width, height, num_time_steps, 2, 2)
    assert m.E.shape == (width, height, num_time_steps, 2, 2)
    assert m.principal_strain.shape == (width, height, num_time_steps, 2)


if __name__ == "__main__":
    # main()
    # mech_obj()
    # test_shapes()
    test_dx(0.5)
