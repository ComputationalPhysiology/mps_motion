from pathlib import Path

import ap_features as apf
import cv2
import dask.array as da
import matplotlib.pyplot as plt
import mps
import numpy as np
import tqdm

from mps_motion_tracking import frame_sequence as fs
from mps_motion_tracking import lucas_kanade, utils

here = Path(__file__).absolute().parent


def main():

    data = utils.MPSData(
        **np.load(
            here.joinpath("../../datasets/mps_data.npy"), allow_pickle=True
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
    breakpoint()
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
        u_norm, data.time_stamps, pacing=data.pacing, correct_background=True
    )
    beats = trace.chop(ignore_pacing=True)
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(trace.t, trace.y)

    for beat in beats:
        ax[0, 1].plot(beat.t, beat.y)

    for beat in beats:
        ax[1, 0].plot(beat.t - beat.t[0], beat.y)

    avg_beat = trace.average_beat()
    ax[1, 1].plot(avg_beat.t, avg_beat.y)
    fig.savefig("displacement_norm.png")


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
        "lukas_kanade_displacement_flow.mp4", fourcc, fps, (width, height)
    )
    out_hsv = cv2.VideoWriter(
        "lukas_kanade_displacement_hsv.mp4", fourcc, fps, (width, height)
    )

    for i in tqdm.tqdm(range(data.num_frames)):
        im = utils.to_uint8(data.frames[:, :, i])
        flow = disp[:, :, :, i]
        out_flow.write(utils.draw_flow(im, flow))
        out_hsv.write(utils.draw_hsv(flow))

    out_flow.release()
    out_hsv.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # main()
    # postprocess_displacement()
    plot_saved_displacement()
