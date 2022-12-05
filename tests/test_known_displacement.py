from collections import namedtuple
from pathlib import Path

import mps_motion as mt
import numpy as np
import pytest
from scipy.ndimage import geometric_transform

HERE = Path(__file__).parent.absolute()

SyntData = namedtuple("SyntData", ["data", "mech"])


def get_func(a=0.005, b=0.007):
    def func(x):
        return (x[0] * (1 - a), x[1] * (1 - b))

    return func


@pytest.fixture(scope="session", params=[1, 0.325])
def gen_data(request):
    frame1 = np.load(HERE.joinpath("first_frame.npy"))
    func1 = get_func(a=0.01, b=0.02)
    frame2 = geometric_transform(frame1.T, func1).T
    func2 = get_func(a=0.02, b=0.01)
    frame3 = geometric_transform(frame1.T, func2).T
    frames = np.stack([frame1, frame2, frame3], axis=-1)

    time_stamps = np.array([0, 1, 2])

    info = {
        "size_x": frames.shape[0],
        "size_y": frames.shape[1],
        "num_frames": frames.shape[2],
        "um_per_pixel": request.param,
    }

    data = mt.utils.MPSData(
        frames=frames,
        time_stamps=time_stamps,
        info=info,
        metadata={},
    )
    x, y = np.meshgrid(np.arange(frame1.shape[1]), np.arange(frame1.shape[0]))

    u01, v01 = func1([x, y])
    u_exact1 = x - u01
    v_exact1 = y - v01
    u1 = np.stack([u_exact1, v_exact1], axis=-1)

    u02, v02 = func2([x, y])
    u_exact2 = x - u02
    v_exact2 = y - v02
    u2 = np.stack([u_exact2, v_exact2], axis=-1)

    u0 = np.zeros_like(u1)
    u = mt.VectorFrameSequence(np.stack([u0, u1, u2], axis=2))

    mech = mt.Mechanics(u, time_stamps)

    return SyntData(data=data, mech=mech)


def test_synthetic_strain(gen_data):
    E = gen_data.mech.E
    du = gen_data.mech.du
    F = gen_data.mech.F

    assert (E[:, :, 0, :, :] == 0).all()

    a = 0
    b = 0.02
    c = 0.01
    d = 0

    assert (np.abs(du[:, :, 1, 0, 0] - a) < 1e-12).all()
    assert (np.abs(du[:, :, 1, 0, 1] - b) < 1e-12).all()
    assert (np.abs(du[:, :, 1, 1, 0] - c) < 1e-12).all()
    assert (np.abs(du[:, :, 1, 1, 1] - d) < 1e-12).all()

    assert (np.abs(F[:, :, 1, 0, 0] - (1 + a)) < 1e-12).all()
    assert (np.abs(F[:, :, 1, 0, 1] - b) < 1e-12).all()
    assert (np.abs(F[:, :, 1, 1, 0] - c) < 1e-12).all()
    assert (np.abs(F[:, :, 1, 1, 1] - (1 + d)) < 1e-12).all()

    assert (np.abs(E[:, :, 1, 0, 0] - 0.5 * ((1 + a) ** 2 + c**2 - 1)) < 1e-12).all()
    assert (np.abs(E[:, :, 1, 0, 1] - 0.5 * (b * (1 + a) + c * (1 + d))) < 1e-12).all()
    assert (np.abs(E[:, :, 1, 1, 0] - 0.5 * (b * (1 + a) + c * (1 + d))) < 1e-12).all()
    assert (np.abs(E[:, :, 1, 1, 1] - 0.5 * ((1 + d) ** 2 + b**2 - 1)) < 1e-12).all()


@pytest.mark.parametrize(
    "flow, tol",
    [
        (mt.farneback.flow, 0.05),
        (mt.lucas_kanade.flow, 0.21),
        (mt.block_matching.flow, 0.9),
        (mt.dualtvl1.flow, 0.05),
    ],
)
def test_synthetic_flow(flow, tol, gen_data):
    frame1 = gen_data.data.frames[:, :, 0]
    frame2 = gen_data.data.frames[:, :, 1]

    u_exact = gen_data.mech.u[:, :, 1, 0]
    v_exact = gen_data.mech.u[:, :, 1, 1]

    U = flow(frame2, frame1)

    u = U[:, :, 0]
    v = U[:, :, 1]

    assert np.linalg.norm(u - u_exact) / np.linalg.norm(u_exact) < tol
    assert np.linalg.norm(v - v_exact) / np.linalg.norm(v_exact) < tol


@pytest.mark.parametrize("unit", ["um", "pixels"])
def test_synthetic_pixels(unit, gen_data):
    data = gen_data.data
    u_exact = gen_data.mech.u

    opt_flow = mt.OpticalFlow(
        data,
        mt.FLOW_ALGORITHMS.farneback,
    )

    u = opt_flow.get_displacements(
        recompute=True,
        unit=unit,
        reference_frame=0,
        smooth_ref_transition=False,
    )

    u_exact_norm = u_exact.norm()
    u_norm = u.norm()

    # Exact solution is zero so lets do the mean
    assert np.abs((u_exact_norm - u_norm)[:, :, 0]).mean().compute() < 0.001
    fac = 1 if unit == "pixels" else data.info["um_per_pixel"]

    assert (
        np.linalg.norm((u_exact_norm * fac - u_norm)[:, :, 1])
        / np.linalg.norm(u_exact_norm[:, :, 1] * fac)
    ).compute() < 0.1

    assert (
        np.linalg.norm((u_exact_norm * fac - u_norm)[:, :, 2])
        / np.linalg.norm(u_exact_norm[:, :, 2] * fac)
    ).compute() < 0.1

    mech = mt.Mechanics(u)

    dE = (gen_data.mech.E - mech.E).norm()

    assert np.abs(dE[:, :, 0]).mean().compute() < 0.01
    assert np.abs(dE[:, :, 1]).mean().compute() < 0.05
    assert np.abs(dE[:, :, 2]).mean().compute() < 0.05
