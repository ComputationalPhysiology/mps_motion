from pathlib import Path

import flowiz

"""
https://github.com/tsenst/CrowdFlow


https://vision.middlebury.edu/flow/data/

https://github.com/opencv/opencv_contrib/blob/master/modules/optflow/samples/optical_flow_benchmark.py

"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

from mps_motion_tracking import dualtvl10, farneback, lucas_kanade


def dimetrodon():
    folder = Path("../datasets/Dimetrodon")
    frames = []
    for filename in ["frame10.png", "frame11.png"]:

        image = cv2.imread(folder.joinpath(filename).as_posix())
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

    flowfile = folder.joinpath("flow10.flo")
    true_flow = flowiz.read_flow(flowfile.as_posix())
    tf = np.swapaxes(np.array(flowiz.flowiz._normalize_flow(true_flow)).T, 0, 1)

    dual_flow = dualtvl10.flow(frames[1], frames[0])
    farneback_flow = farneback.flow(
        frames[1],
        frames[0],
        winsize=15,
    )
    points = lucas_kanade.get_uniform_reference_points(frames[0], step=1)
    lk_flow = np.swapaxes(
        lucas_kanade.flow(frames[1], frames[0], points).reshape(
            dual_flow.shape[1], dual_flow.shape[0], 2
        ),
        0,
        1,
    )

    # bm_flow = block_matching.flow(frames[0], frames[1])

    vmin = 0
    vmax = 255

    fig, ax = plt.subplots(2, 2)  # , sharex=True, sharey=True)
    ax[0, 0].imshow(flowiz.convert_from_flow(tf), vmin=vmin, vmax=vmax)
    ax[0, 0].set_title("True flow")

    ax[0, 1].imshow(flowiz.convert_from_flow(dual_flow), vmin=vmin, vmax=vmax)
    ax[0, 1].set_title("dualtvl10")

    ax[1, 0].imshow(flowiz.convert_from_flow(farneback_flow), vmin=vmin, vmax=vmax)
    ax[1, 0].set_title("farneback")

    ax[1, 1].imshow(flowiz.convert_from_flow(lk_flow), vmin=vmin, vmax=vmax)
    ax[1, 1].set_title("lucas kanade")

    fig, ax = plt.subplots(4, 2)  # , sharex=True, sharey=True)

    uv = flowiz.convert_from_flow(tf, mode="UV")
    ax[0, 0].set_title("Horizontal Flow (U) - True flow")
    ax[0, 0].imshow(uv[..., 0], cmap=plt.get_cmap("binary"), vmin=vmin, vmax=vmax)
    ax[0, 1].set_title("Vertical Flow (V) - True flow")
    ax[0, 1].imshow(uv[..., 1], cmap=plt.get_cmap("binary"), vmin=vmin, vmax=vmax)

    uv = flowiz.convert_from_flow(dual_flow, mode="UV")
    ax[1, 0].set_title("Horizontal Flow (U) - dualtv10")
    ax[1, 0].imshow(uv[..., 0], cmap=plt.get_cmap("binary"), vmin=vmin, vmax=vmax)
    ax[1, 1].set_title("Vertical Flow (V) - dualtvl10")
    ax[1, 1].imshow(uv[..., 1], cmap=plt.get_cmap("binary"), vmin=vmin, vmax=vmax)

    uv = flowiz.convert_from_flow(farneback_flow, mode="UV")
    ax[2, 0].set_title("Horizontal Flow (U) - farenback")
    ax[2, 0].imshow(uv[..., 0], cmap=plt.get_cmap("binary"), vmin=vmin, vmax=vmax)
    ax[2, 1].set_title("Vertical Flow (V) - farenback")
    ax[2, 1].imshow(uv[..., 1], cmap=plt.get_cmap("binary"), vmin=vmin, vmax=vmax)

    uv = flowiz.convert_from_flow(lk_flow, mode="UV")
    ax[3, 0].set_title("Horizontal Flow (U) - lucas kanade")
    ax[3, 0].imshow(uv[..., 0], cmap=plt.get_cmap("binary"), vmin=vmin, vmax=vmax)
    ax[3, 1].set_title("Vertical Flow (V) - lucas kanade")
    ax[3, 1].imshow(uv[..., 1], cmap=plt.get_cmap("binary"), vmin=vmin, vmax=vmax)

    fig.tight_layout()
    plt.show()

    # flow = read_flow(flowfile.as_posix())
    # np.save(folder.joinpath("flow.npy"), flow)
    # np.save(folder.joinpath("data.npy"), np.array(frames))

    # from IPython import embed

    # embed()
    # exit()


if __name__ == "__main__":
    dimetrodon()
