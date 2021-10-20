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
            filter_kernel_size=5,
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

    data = mps.MPS("../PointH4A_ChannelBF_VC_Seq0018.nd2")
    path = Path("v_norm.npy")
    if not path.is_file():
        disp = farneback.get_displacements(data.frames, data.frames[:, :, 0])
        u = fs.VectorFrameSequence(disp)
        u_th = u.threshold_norm(0, 10)
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

    data = mps.MPS("../PointH4A_ChannelBF_VC_Seq0018.nd2")
    path = Path("disp.npy")
    if not path.is_file():
        disp = farneback.get_displacements(
            data.frames,
            data.frames[:, :, 0],
            filter_kernel_size=3,
        )
        np.save(path, disp.compute())
    disp = da.from_array(np.load(path, mmap_mode="r"))
    u = fs.VectorFrameSequence(disp)

    # mech = mechanics.Mechancis(u=u, t=data.time_stamps)
    # Exx = mech.E.x.threshold(-0.2, 0.2)
    # visu.heatmap("heatmap_Exx.mp4", data=Exx, fps=data.framerate)

    visu.heatmap(
        "heatmap_u_norm.mp4",
        data=u.norm().threshold(0, 10),
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
    plot_velocity()
    # create_velocity_flow_field()
    # create_heatmap()
    # main()
