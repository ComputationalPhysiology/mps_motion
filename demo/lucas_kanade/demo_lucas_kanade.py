from pathlib import Path

import ap_features as apf
import cv2
import dask.array as da
import matplotlib.pyplot as plt
import mps
import numpy as np
import tqdm

from mps_motion_tracking import frame_sequence as fs
from mps_motion_tracking import lucas_kanade
from mps_motion_tracking import mechanics
from mps_motion_tracking import utils
from mps_motion_tracking import visu

here = Path(__file__).absolute().parent


def main():

    data = utils.MPSData(
        **np.load(
            here.joinpath("../../datasets/mps_data.npy"),
            allow_pickle=True,
        ).item()
    )
    data = mps.MPS("../PointH4A_ChannelBF_VC_Seq0018.nd2")

    # m = lucas_kanade.flow(data.frames[:, :, 0], data.frames[:, :, 1])

    # disp, ref_points = lucas_kanade.get_displacements(
    #     data.frames, data.frames[:, :, 0], step=8
    # )
    disp = lucas_kanade.get_displacements(
        data.frames,
        data.frames[:, :, 0],
        step=48,
    )

    print("Convert to dask array")
    U = da.from_array(np.swapaxes(disp, 2, 3))

    u = fs.VectorFrameSequence(U)
    print("Compute norm")
    u_norm = u.norm().mean().compute()
    print("Plot")
    plt.plot(data.time_stamps, u_norm)
    plt.show()


def plot_saved_displacement():

    data = mps.MPS("../PointH4A_ChannelBF_VC_Seq0018.nd2")
    disp = np.load("disp.npy", mmap_mode="r")

    U = da.from_array(np.swapaxes(disp, 2, 3))

    u = fs.VectorFrameSequence(U)
    print("Compute norm")
    u_norm = u.norm().mean().compute()

    print("Plot")
    trace = apf.Beats(
        u_norm,
        data.time_stamps,
        pacing=data.pacing,
        background_correction_method="subtract",
        zero_index=60,
    )
    beats = trace.chop(ignore_pacing=True)
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
    disp = np.load("disp.npy", mmap_mode="r")

    U = da.from_array(np.swapaxes(disp, 2, 3))
    V = mechanics.compute_velocity(U, data.time_stamps)

    v = fs.VectorFrameSequence(V)
    # print("Compute norm")
    v_norm = v.norm().mean().compute()
    trace = apf.Beats(
        v_norm,
        data.time_stamps[:-1],
        pacing=data.pacing,
        background_correction_method="subtract",
        # zero_index=60,
    )
    beats = trace.chop(ignore_pacing=True)
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


def postprocess_displacement():
    # data = utils.MPSData(
    #     **np.load(
    #         here.joinpath("../../datasets/mps_data.npy"), allow_pickle=True
    #     ).item()
    # )
    data = mps.MPS("../PointH4A_ChannelBF_VC_Seq0018.nd2")
    disp = lucas_kanade.get_displacements(
        data.frames,
        data.frames[:, :, 0],
        step=48,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    width = data.size_y
    height = data.size_x
    fps = data.framerate

    out_flow = cv2.VideoWriter(
        "lukas_kanade_displacement_flow.mp4",
        fourcc,
        fps,
        (width, height),
    )
    out_hsv = cv2.VideoWriter(
        "lukas_kanade_displacement_hsv.mp4",
        fourcc,
        fps,
        (width, height),
    )

    for i in tqdm.tqdm(range(data.num_frames)):
        im = utils.to_uint8(data.frames[:, :, i])
        flow = disp[:, :, :, i]
        out_flow.write(utils.draw_flow(im, flow))
        out_hsv.write(utils.draw_hsv(flow))

    out_flow.release()
    out_hsv.release()
    cv2.destroyAllWindows()


def create_flow_field():

    data = mps.MPS("../PointH4A_ChannelBF_VC_Seq0018.nd2")
    # disp = lucas_kanade.get_displacements(
    #     data.frames,
    #     data.frames[:, :, 0],
    #     step=48,
    # )
    disp = np.load("disp.npy", mmap_mode="r")

    visu.quiver_video(data, disp, "flow.mp4", step=48, scale=10)
    # visu.hsv_video(data, disp, "hsv.mp4")


def create_velocity_flow_field():

    data = mps.MPS("../PointH4A_ChannelBF_VC_Seq0018.nd2")
    # disp = lucas_kanade.get_displacements(
    #     data.frames,
    #     data.frames[:, :, 0],
    #     step=48,
    # )
    disp = np.load("disp.npy", mmap_mode="r")
    V = mechanics.compute_velocity(disp, data.time_stamps)

    visu.quiver_video(data, V, "velocity_flow.mp4", step=48, scale=500, velocity=True)


if __name__ == "__main__":
    # main()
    # postprocess_displacement()
    # plot_saved_displacement()
    create_flow_field()
    # plot_velocity()
    # create_velocity_flow_field()
