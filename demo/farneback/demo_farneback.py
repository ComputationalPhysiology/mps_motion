from pathlib import Path

import ap_features as apf
import dask.array as da
import matplotlib.pyplot as plt
import mps
import numpy as np

import mps_motion_tracking as mmt
from mps_motion_tracking import farneback
from mps_motion_tracking import frame_sequence as fs
from mps_motion_tracking import mechanics
from mps_motion_tracking import visu

here = Path(__file__).absolute().parent


def plot_displacement():

    data = mps.MPS("../PointH4A_ChannelBF_VC_Seq0018.nd2")
    path = Path("u_norm.npy")
    if not path.is_file():
        disp = farneback.get_displacements(
            data.frames,
            data.frames[:, :, 0],
        )
        u = fs.VectorFrameSequence(disp)
        u_norm = u.norm().threshold(0, 10).mean().compute()
        np.save(path, u_norm)
    u_norm = np.load(path)

    print("Plot")
    trace = apf.Beats(
        u_norm,
        data.time_stamps,
        pacing=data.pacing,
        background_correction_method="subtract",
        chopping_options={"ignore_pacing": True},
    )
    beats = trace.beats
    fig, ax = plt.subplots(3, 2)
    ax[0, 0].plot(trace.t, trace.y)

    for beat in beats:
        ax[0, 1].plot(beat.t, beat.y)

    for beat in beats:
        ax[1, 0].plot(beat.t - beat.t[0], beat.y)

    avg_beat = trace.average_beat()
    ax[1, 1].plot(avg_beat.t, avg_beat.y)
    ax[2, 0].plot(trace.t, trace.original_y)
    ax[2, 0].plot(trace.t, trace.background.background)
    fig.savefig("displacement_norm.png")


def plot_velocity():

    # data = mps.MPS("../PointH4A_ChannelBF_VC_Seq0018.nd2")
    data = mmt.scaling.resize_data(
        mps.MPS("../PointH4A_ChannelBF_VC_Seq0018.nd2"),
        scale=0.4,
    )
    path = Path("v_norm.npy")
    if not path.is_file():
        disp = farneback.get_displacements(
            data.frames,
            data.frames[:, :, 0],
            filter_kernel_size=15,
        )
        u = fs.VectorFrameSequence(disp * (1 / 0.4))
        # u_th = u.threshold_norm(0, 6)
        u_th = u
        V = mechanics.compute_velocity(u_th.array, data.time_stamps)

        v = fs.VectorFrameSequence(V)
        # print("Compute norm")
        v_norm = v.norm().mean().compute()
        np.save(path, v_norm)
    v_norm = np.load(path)
    trace = apf.Beats(
        v_norm,
        data.time_stamps[:-1],
        pacing=data.pacing[:-1],
        background_correction_method="subtract",
        chopping_options={"ignore_pacing": True},
    )

    beats = trace.beats
    fig, ax = plt.subplots(3, 2)
    ax[0, 0].plot(trace.t, trace.y)

    for beat in beats:
        ax[0, 1].plot(beat.t, beat.y)

    for beat in beats:
        ax[1, 0].plot(beat.t - beat.t[0], beat.y)

    avg_beat = trace.average_beat()
    ax[1, 1].plot(avg_beat.t, avg_beat.y)
    ax[2, 0].plot(trace.t, trace.original_y)
    ax[2, 0].plot(trace.t, trace.background.background)
    fig.savefig("velocity_norm.png")


def create_heatmap():

    # data = mps.MPS("../PointH4A_ChannelBF_VC_Seq0018.nd2")
    data = mmt.scaling.resize_data(
        mps.MPS("../PointH4A_ChannelBF_VC_Seq0018.nd2"),
        scale=0.4,
    )
    path = Path("disp.npy")
    if not path.is_file():
        disp = farneback.get_displacements(
            data.frames,
            data.frames[:, :, 0],
        )
        np.save(path, disp.compute())
    disp = da.from_array(np.load(path, mmap_mode="r"))
    u = fs.VectorFrameSequence(disp)

    # mech = mechanics.Mechanics(u=u, t=data.time_stamps)
    # Exx = mech.E.x.threshold(-0.2, 0.2)
    # visu.heatmap("heatmap_Exx.mp4", data=Exx, fps=data.framerate)

    visu.heatmap(
        "heatmap_Exx_spline.mp4",
        data=u.norm(),
        fps=data.framerate,
        cmap="inferno",
        transpose=True,
    )


def compare_strain_disp_spline():

    data = mmt.scaling.resize_data(
        mps.MPS("../PointH4A_ChannelBF_VC_Seq0018.nd2"),
        scale=0.4,
    )
    path = Path("disp.npy")
    if not path.is_file():
        disp = farneback.get_displacements(
            data.frames,
            data.frames[:, :, 0],
        )
        np.save(path, disp.compute())
    disp = da.from_array(np.load(path, mmap_mode="r"))
    u = fs.VectorFrameSequence(disp)
    U = u.spline_smooth()

    # x = np.arange(u.shape[0])
    # y = np.arange(u.shape[1])
    # Y, X = np.meshgrid(y, x)
    # u_max = u[:, :, 243, :].compute()
    # new_u = mmt.filters.spline_smooth_u((u_max[:, :, 0], u_max[:, :, 1], X, Y))

    # fig, ax = plt.subplots(2, 2)
    # vmin = 0
    # vmax = u_max.max()
    # ax[0, 0].imshow(u_max[:, :, 0], vmin=vmin, vmax=vmax)
    # ax[0, 0].set_title("Original X")
    # ax[0, 1].imshow(u_max[:, :, 1], vmin=vmin, vmax=vmax)
    # ax[0, 1].set_title("Original Y")

    # ax[1, 0].imshow(new_u[0], vmin=vmin, vmax=vmax)
    # ax[1, 0].set_title("Interpolated X")
    # ax[1, 1].imshow(new_u[1], vmin=vmin, vmax=vmax)
    # ax[1, 1].set_title("Interpolated Y")
    # plt.show()

    # breakpoint()
    # arr = mmt.filters.spline_smooth(u.array)
    # U = fs.VectorFrameSequence(da.from_array(arr), scale=u.scale, dx=u.dx)

    visu.heatmap(
        "heatmap_u.mp4",
        data=u.norm(),
        fps=data.framerate,
        cmap="inferno",
        transpose=True,
    )

    visu.heatmap(
        "heatmap_u_spline.mp4",
        data=U.norm(),
        fps=data.framerate,
        cmap="inferno",
        transpose=True,
    )

    mech = mechanics.Mechanics(u=u, t=data.time_stamps)
    mech_spline = mechanics.Mechanics(u=U, t=data.time_stamps)

    u_norm = u.norm().mean().compute()
    u_norm_spline = U.norm().mean().compute()

    # x0 = u.shape[0] // 2 - 50
    # x1 = u.shape[0] // 2 + 50
    # y0 = u.shape[1] // 2 - 50
    # y1 = u.shape[1] // 2 + 50

    Exx = mech.E.x.mean().compute()
    Exx_spline = mech_spline.E.x.mean().compute()
    # Exx_spline2 = mech_spline.E.x[x0:x1, y0:y1, :].mean((0, 1)).compute()
    Eyy = mech.E.y.mean().compute()
    Eyy_spline = mech_spline.E.y.mean().compute()
    # Eyy_spline2 = mech_spline.E.y[x0:x1, y0:y1, :].mean((0, 1)).compute()
    Exy = mech.E.xy.mean().compute()
    Exy_spline = mech_spline.E.xy.mean().compute()
    # Exy_spline2 = mech_spline.E.xy[x0:x1, y0:y1, :].mean((0, 1)).compute()

    fig, ax = plt.subplots(2, 2, sharex=True)
    (l1,) = ax[0, 0].plot(data.time_stamps, u_norm)
    (l2,) = ax[0, 0].plot(data.time_stamps, u_norm_spline)
    ax[0, 0].set_title("Displacement")

    ax[0, 1].plot(data.time_stamps, Exx)
    ax[0, 1].plot(data.time_stamps, Exx_spline)
    ax[0, 1].plot(data.time_stamps, Exx_spline)
    ax[0, 1].set_title("Exx")

    ax[1, 0].plot(data.time_stamps, Eyy)
    ax[1, 0].plot(data.time_stamps, Eyy_spline)
    ax[1, 0].set_title("Eyy")

    ax[1, 1].plot(data.time_stamps, Exy)
    ax[1, 1].plot(data.time_stamps, Exy_spline)
    ax[1, 1].set_title("Exy")
    plt.show()
    # # fig.legend((l1, l2), ("Original", "Spline"), loc=)

    visu.heatmap(
        "heatmap_Exx.mp4",
        data=mech.E.x,
        fps=data.framerate,
        cmap="inferno",
        transpose=True,
    )
    visu.heatmap(
        "heatmap_Eyy.mp4",
        data=mech.E.y,
        fps=data.framerate,
        cmap="inferno",
        transpose=True,
    )
    visu.heatmap(
        "heatmap_Exx_spline.mp4",
        data=mech_spline.E.x,
        fps=data.framerate,
        cmap="inferno",
        transpose=True,
    )
    visu.heatmap(
        "heatmap_Eyy_spline.mp4",
        data=mech_spline.E.y,
        fps=data.framerate,
        cmap="inferno",
        transpose=True,
    )


def create_flow_field():

    data = mps.MPS("../PointH4A_ChannelBF_VC_Seq0018.nd2")
    disp = farneback.get_displacements(
        data.frames,
        data.frames[:, :, 0],
        filter_kernel_size=3,
    )
    np.save("disp.npy", disp)
    visu.quiver_video(data, disp, "flow.mp4", step=48, vector_scale=10)


def create_velocity_flow_field():

    data = mmt.scaling.resize_data(
        mps.MPS("../PointH4A_ChannelBF_VC_Seq0018.nd2"),
        scale=0.4,
    )
    opt_flow = mmt.OpticalFlow(data, flow_algorithm="farneback", reference_frame=0)

    u = opt_flow.get_displacements()
    # V = mechanics.compute_velocity(u.array, data.time_stamps)
    # vel = fs.VectorFrameSequence(V)
    angle = u.angle().array.compute()
    angle *= 180 / np.pi
    x, y = np.meshgrid(
        np.arange(u.shape[1]),
        np.arange(u.shape[0]),
    )
    index = 83
    plt.imshow(data.frames[:, :, index], cmap="gray")
    N = 10
    plt.quiver(
        x[::N, ::N],
        y[::N, ::N],
        u.array[::N, ::N, index, 0],
        -u.array[::N, ::N, index, 1],
        angle[::N, ::N, index],
        scale_units="inches",
        scale=10,
        cmap="twilight",
        clim=(-180, 180),
    )
    # plt.imshow(angle[:, :, 71], cmap="twilight", vmin=-180, vmax=180)
    plt.colorbar()
    plt.show()
    # breakpoint()

    # u_norm = u.norm().mean().compute()
    # vel_norm = vel.norm().mean().compute() * 1000
    # fig, ax = plt.subplots()
    # (l1,) = ax.plot(data.time_stamps, u_norm)
    # ax2 = ax.twinx()
    # (l2,) = ax2.plot(data.time_stamps[:-1], vel_norm, color="r")
    # ax.legend([l1, l2], ["displacement", "velocity"], loc="best")
    # ax.set_xlabel("Time [ms]")
    # ax.set_ylabel("Displacement [\u00B5m")
    # ax2.set_ylabel("Displacement [\u00B5m / s")
    # fig.savefig("disp_velocity.png")
    # plt.show()


def main():
    shape = (3, 3, 2)

    import dask.array as da

    x = da.random.random(shape)
    x_norm = da.linalg.norm(x, axis=-1)

    indices = da.stack([x_norm < 0.5, x_norm < 0.5], axis=-1).flatten().compute()
    print(indices.shape)
    print(x.flatten()[indices])


if __name__ == "__main__":
    # main()
    # postprocess_displacement()
    # plot_displacement()
    # create_flow_field()
    # plot_velocity()
    # create_velocity_flow_field()
    # create_heatmap()
    # main()
    compare_strain_disp_spline()
