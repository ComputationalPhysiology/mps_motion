from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from mps_motion_tracking import farneback
from mps_motion_tracking import Mechancis
from mps_motion_tracking import OpticalFlow
from mps_motion_tracking import utils

here = Path(__file__).absolute().parent


def main():

    data = utils.MPSData(
        **np.load(
            here.joinpath("../../datasets/mps_data.npy"),
            allow_pickle=True,
        ).item()
    )

    disp = farneback.get_displacements(data.frames, data.frames[:, :, 0])
    np.save("farneback_disp.npy", disp)
    vel = farneback.get_velocities(data.frames, data.frames[:, :, 0])
    np.save("farneback_vel.npy", vel)


def postprocess_displacement():
    data = utils.MPSData(
        **np.load(
            here.joinpath("../../datasets/mps_data.npy"),
            allow_pickle=True,
        ).item()
    )
    disp = np.load("farneback_disp.npy")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    width = data.size_y
    height = data.size_x
    fps = data.framerate

    out_flow = cv2.VideoWriter(
        "farneback_displacement_flow.mp4",
        fourcc,
        fps,
        (width, height),
    )
    out_hsv = cv2.VideoWriter(
        "farneback_displacement_hsv.mp4",
        fourcc,
        fps,
        (width, height),
    )

    for i in tqdm.tqdm(range(data.num_frames)):
        im = utils.to_uint8(data.frames[:, :, i])
        flow = disp[:, :, :, i]
        # out.write(im)
        out_flow.write(utils.draw_flow(im, flow))
        out_hsv.write(utils.draw_hsv(flow))
        # cv2.imshow("flow", utils.draw_flow(im, flow))

        key = cv2.waitKey(1)
        if key == 27:
            break

    out_flow.release()
    out_hsv.release()
    cv2.destroyAllWindows()


def postprocess_velocities():
    data = utils.MPSData(
        **np.load(
            here.joinpath("../../datasets/mps_data.npy"),
            allow_pickle=True,
        ).item()
    )
    disp = np.load("farneback_vel.npy")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    width = data.size_y
    height = data.size_x
    fps = data.framerate

    out_flow = cv2.VideoWriter(
        "farneback_velocity_flow.mp4",
        fourcc,
        fps,
        (width, height),
    )
    out_hsv = cv2.VideoWriter(
        "farneback_velocity_hsv.mp4",
        fourcc,
        fps,
        (width, height),
    )

    for i in tqdm.tqdm(range(1, data.num_frames)):
        im = utils.to_uint8(data.frames[:, :, i])
        flow = disp[:, :, :, i]
        # out.write(im)
        out_flow.write(utils.draw_flow(im, flow))
        out_hsv.write(utils.draw_hsv(flow))
        # cv2.imshow("flow", utils.draw_flow(im, flow))

        key = cv2.waitKey(1)
        if key == 27:
            break

    out_flow.release()
    out_hsv.release()
    cv2.destroyAllWindows()


def plot_displacements():

    data = utils.MPSData(
        **np.load(
            here.joinpath("../../datasets/mps_data.npy"),
            allow_pickle=True,
        ).item()
    )
    disp = np.load("farneback_disp.npy") * data.info["um_per_pixel"]

    import dask.array as da

    d = da.from_array(disp)
    disp_norm = da.linalg.norm(d, axis=2).compute()
    u_mean_pixel = disp_norm.mean((0, 1))

    u_mean_um = u_mean_pixel

    fig, ax = plt.subplots()
    ax.plot(data.time_stamps, u_mean_um)
    ax.set_title("Mean displacement")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Magnitude of displacement [um]")
    fig.savefig("farneback_mean_displacement.png")

    ux_mean_pixel = disp[:, :, 0, :].mean((0, 1))
    ux_mean_um = ux_mean_pixel * data.info["um_per_pixel"]
    uy_mean_pixel = disp[:, :, 1, :].mean((0, 1))
    uy_mean_um = uy_mean_pixel * data.info["um_per_pixel"]
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].plot(data.time_stamps, ux_mean_um)
    ax[0].set_title("Mean X-displacement")
    ax[0].set_xlabel("Time [ms]")
    ax[0].set_ylabel("Displacement [um]")

    ax[1].plot(data.time_stamps, uy_mean_um)
    ax[1].set_title("Mean y-displacement")
    ax[1].set_xlabel("Time [ms]")
    fig.savefig("farneback_mean_displacement_comp.png")

    max_disp_x = np.abs(disp[:, :, 0, :]).max(-1)
    max_disp_y = np.abs(disp[:, :, 1, :]).max(-1)
    max_disp = disp_norm.max(-1)

    vmin = 0
    vmax = max_disp.max()
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    # Norm
    ax[0].imshow(max_disp, vmin=vmin, vmax=vmax)
    ax[0].set_title("Max displacement norm")
    # X
    ax[1].imshow(max_disp_x, vmin=vmin, vmax=vmax)
    ax[1].set_title("Max displacement X")
    # Y
    im = ax[2].imshow(max_disp_y, vmin=vmin, vmax=vmax)

    ax[2].set_title("Max displacement Y")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("um")
    fig.savefig("farneback_max_displacement.png")

    # plt.show()


def plot_strain():

    # data = utils.MPSData(
    #     **np.load(
    #         here.joinpath("../../datasets/mps_data.npy"), allow_pickle=True
    #     ).item()
    # )
    import mps

    data = mps.MPS(here.parent.joinpath("PointH4A_ChannelBF_VC_Seq0018.nd2"))

    flow = OpticalFlow(data, "farneback")
    u = flow.get_displacements(scale=0.3)
    mech = Mechancis(u)

    print(mech)
    from IPython import embed

    embed()
    exit()


if __name__ == "__main__":
    # main()
    # postprocess_displacement()
    # postprocess_velocities()
    # plot_displacements()
    plot_strain()
