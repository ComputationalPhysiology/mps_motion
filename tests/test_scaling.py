import numpy as np
from mps_motion import Mechanics
from mps_motion import OpticalFlow
from mps_motion import scaling
from mps_motion import utils


def test_resize_data():
    frames = np.ones((100, 100, 3))
    times = np.linspace(0, 5000, frames.shape[-1])
    info = dict(um_per_pixel=1.0)
    data = utils.MPSData(frames=frames, time_stamps=times, info=info)
    scale = 0.5
    scaled_data = scaling.resize_data(data, scale)
    assert scaled_data.info["um_per_pixel"] == info["um_per_pixel"] / scale
    assert scaled_data.frames.shape == (50, 50, 3)
    assert np.isclose(scaled_data.time_stamps, times).all()


def test_resize_frames_displacement():
    width = 10
    height = 15
    num_time_points = 4
    scale = 0.4

    np.random.seed(1)
    u = np.zeros((height, width, 2, num_time_points))
    u[:, :, 0, 0] = np.fromfunction(
        lambda y, x: x / width,
        shape=(height, width),
        dtype=float,
    )
    u[:, :, 1, 0] = np.fromfunction(
        lambda y, x: y / height,
        shape=(height, width),
        dtype=float,
    )
    u_x_resized = scaling.resize_frames(u[:, :, 0, :], scale=scale)
    u_y_resized = scaling.resize_frames(u[:, :, 1, :], scale=scale)
    u_resized = np.stack((u_x_resized, u_y_resized), axis=2)

    u_x_orig = scaling.resize_frames(u_x_resized, scale=1 / scale)
    u_y_orig = scaling.resize_frames(u_y_resized, scale=1 / scale)
    u_orig = np.stack((u_x_orig, u_y_orig), axis=2)

    assert u_orig.shape == u.shape
    assert abs(u[:, :, 0, 0].mean() - u_resized[:, :, 0, 0].mean()) < 1e-12
    assert abs(u[:, :, 1, 0].mean() - u_resized[:, :, 1, 0].mean()) < 1e-12

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 4)
    # ax[0].imshow(u[:, :, 1, 0])
    # ax[1].imshow(u_resized[:, :, 1, 0])
    # ax[2].imshow(u_orig[:, :, 1, 0])
    # ax[3].imshow((u - u_orig)[:, :, 1, 0])
    # plt.show()


def _test_resize_frames_units():
    """This is a visual test"""
    import matplotlib.pyplot as plt
    import synthetic_data

    frames = synthetic_data.create_circle_data(dx=5, dy=0, Nx=200, Ny=200)
    times = np.linspace(0, 5000, frames.shape[-1])
    info = dict(um_per_pixel=1.0)

    # mps.utils.frames2mp4(frames.T, "synthetic_data")
    data = utils.MPSData(frames=frames, time_stamps=times, info=info)
    # new_data = scaling.resize_data(data, scale=0.5)

    # mps.utils.frames2mp4(scaling.resize_frames(frames, 0.5).T, "synthetic_data_05")
    flow = OpticalFlow(data, "farneback")

    # scale = 1.0
    for scale in [0.5, 0.8, 1.0]:
        flow = OpticalFlow(data, "farneback")
        u = flow.get_displacements(scale=scale)  # , unit="pixels")

        mech = Mechanics(u)

        # import dask.array as da
        U = mech.E.compute_eigenvalues()

        # U = mech.u

        u_norm = U.norm().array  # .compute()
        u_norm_time = U.norm().mean()  # .compute()

        # from IPython import embed

        # embed()
        # exit()

        # u_norm = mech.E.x.array.compute()
        # u_norm_time = mech.E.x.mean().compute()

        fig, ax = plt.subplots()
        ax.plot(times, u_norm_time)
        # continue
        # plt.show()

        from mps.analysis import local_averages

        la = local_averages(
            u_norm,
            data.time_stamps,
            background_correction=False,
            N=int(10 * scale),
        )

        fig, ax = plt.subplots(
            la.shape[0],
            la.shape[1],
            sharex=True,
            sharey=True,
            figsize=(12, 12),
        )
        for i in range(la.shape[0]):
            for j in range(la.shape[1]):
                ax[i, j].plot(data.time_stamps, la[i, j, :])

        # mps.utils.frames2mp4(u_norm.T, "disp_norm")

        u_max = np.max(u_norm, axis=2)
        fig, ax = plt.subplots()
        im = ax.imshow(u_max)
        fig.colorbar(im)
    plt.show()


if __name__ == "__main__":
    # test_ScaledDisplacemet()
    # test_resize_frames_units()
    # pass
    _test_resize_frames_units()
