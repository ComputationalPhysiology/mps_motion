from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from mps_motion_tracking import lucas_kanade, utils

here = Path(__file__).absolute().parent


def main():

    data = utils.MPSData(
        **np.load(
            here.joinpath("../../datasets/mps_data.npy"), allow_pickle=True
        ).item()
    )
    disp, ref_points = lucas_kanade.get_displacements(
        data.frames, data.frames[:, :, 0], step=8
    )
    np.save("lk_disp.npy", disp)
    np.save("lk_ref_points.npy", ref_points)


def postprocess_displacement():
    data = utils.MPSData(
        **np.load(
            here.joinpath("../../datasets/mps_data.npy"), allow_pickle=True
        ).item()
    )
    disp = np.load("lk_disp.npy")
    ref_points = np.load("lk_ref_points.npy")

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
        flow = disp[:, :, i]
        # out.write(im)
        out_flow.write(utils.draw_lk_flow(im, flow, ref_points))
        # out_hsv.write(utils.draw_hsv(flow))
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
            here.joinpath("../../datasets/mps_data.npy"), allow_pickle=True
        ).item()
    )
    disp = np.load("lk_disp.npy") * data.info["um_per_pixel"]
    reference_points = np.load("lk_ref_points.npy")

    u_mean_um = np.linalg.norm(disp, axis=1).mean(0)

    fig, ax = plt.subplots()
    ax.plot(data.time_stamps, u_mean_um)
    ax.set_title("Mean displacement")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Magnitude of displacement [um]")
    fig.savefig("lk_mean_displacement.png")

    ux_mean_pixel = disp[:, 0, :].mean(0)
    ux_mean_um = ux_mean_pixel * data.info["um_per_pixel"]
    uy_mean_pixel = disp[:, 1, :].mean(0)
    uy_mean_um = uy_mean_pixel * data.info["um_per_pixel"]
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].plot(data.time_stamps, ux_mean_um)
    ax[0].set_title("Mean X-displacement")
    ax[0].set_xlabel("Time [ms]")
    ax[0].set_ylabel("Displacement [um]")

    ax[1].plot(data.time_stamps, uy_mean_um)
    ax[1].set_title("Mean y-displacement")
    ax[1].set_xlabel("Time [ms]")
    fig.savefig("lk_mean_displacement_comp.png")
    # disp_full = utils.interpolate_lk_flow(
    #     disp, reference_points, data.size_x, data.size_y
    # )

    # max_disp_x = np.abs(disp_full[:, :, 0, :]).max(-1)
    # max_disp_y = np.abs(disp_full[:, :, 1, :]).max(-1)
    # max_disp = np.linalg.norm(disp_full, axis=2).max(-1)

    # vmin = 0
    # vmax = max_disp.max()
    # fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    # # Norm
    # ax[0].imshow(max_disp, vmin=vmin, vmax=vmax)
    # ax[0].set_title("Max displacement norm")
    # # X
    # ax[1].imshow(max_disp_x, vmin=vmin, vmax=vmax)
    # ax[1].set_title("Max displacement X")
    # # Y
    # im = ax[2].imshow(max_disp_y, vmin=vmin, vmax=vmax)

    # ax[2].set_title("Max displacement Y")

    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # cbar = fig.colorbar(im, cax=cbar_ax)
    # cbar.set_label("um")

    # Alternative
    from scipy.interpolate import griddata

    x, y = np.squeeze(reference_points).T
    grid_x, grid_y = np.meshgrid(np.arange(data.size_x), np.arange(data.size_y))

    # plt.show()

    max_x = np.abs(disp[:, 0, :]).max(-1)
    max_y = np.abs(disp[:, 1, :]).max(-1)
    max_norm = np.linalg.norm(disp, axis=1).max(-1)

    vmin = 0
    vmax = max_norm.max()

    grid_max_x = griddata((x, y), max_x, (grid_y, grid_x), method="linear")
    grid_max_y = griddata((x, y), max_y, (grid_y, grid_x), method="linear")
    grid_max_norm = griddata((x, y), max_norm, (grid_y, grid_x), method="linear")

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    ax[0].contourf(grid_y, grid_x, grid_max_norm, vmin=vmin, vmax=vmax)
    ax[0].set_title("Max displacement norm")
    ax[1].contourf(grid_y, grid_x, grid_max_x, vmin=vmin, vmax=vmax)
    ax[1].set_title("Max displacement X")
    im = ax[2].contourf(grid_y, grid_x, grid_max_y, vmin=vmin, vmax=vmax)
    ax[2].set_title("Max displacement Y")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("um")
    fig.savefig("lk_max_displacement.png")
    plt.close("all")


if __name__ == "__main__":
    main()
    postprocess_displacement()
    plot_displacements()
