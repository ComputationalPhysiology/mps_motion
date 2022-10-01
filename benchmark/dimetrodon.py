"""
https://github.com/tsenst/CrowdFlow


https://vision.middlebury.edu/flow/data/

https://github.com/opencv/opencv_contrib/blob/master/modules/optflow/samples/optical_flow_benchmark.py

"""
from pathlib import Path

import cv2
import flowiz
import matplotlib.pyplot as plt
import numpy as np
from mps_motion import block_matching
from mps_motion import dualtvl1
from mps_motion import farneback
from mps_motion import lucas_kanade


def rubber_whale():

    folder = Path("../datasets/RubberWhale")
    frames = []
    for filename in ["frame10.png", "frame11.png"]:

        image = cv2.imread(folder.joinpath(filename).as_posix())
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

    flowfile = folder.joinpath("flow10.flo")
    true_flow = flowiz.read_flow(flowfile.as_posix())
    tf = np.swapaxes(np.array(flowiz.flowiz._normalize_flow(true_flow)).T, 0, 1)

    main(tf, frames)


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
    main(tf, frames)


def main(tf, frames):

    dual_flow = dualtvl1.flow(frames[1], frames[0])
    dual_flow_norm = np.linalg.norm(dual_flow, axis=2)
    dual_flow_norm /= np.nanmax(dual_flow_norm)

    farneback_flow = farneback.flow(
        frames[1],
        frames[0],
    )
    farneback_flow_norm = np.linalg.norm(farneback_flow, axis=2)
    farneback_flow_norm /= farneback_flow_norm.max()

    points = lucas_kanade.get_uniform_reference_points(frames[0], step=4)

    lk_flow = lucas_kanade.flow(frames[1], frames[0], points)
    lk_flow_norm = np.linalg.norm(lk_flow, axis=2)
    lk_flow_norm /= lk_flow_norm.max()

    bm_flow = block_matching.flow(frames[1], frames[0], resize=True)
    bm_flow_norm = np.linalg.norm(bm_flow, axis=2)
    bm_flow_norm /= bm_flow_norm.max()

    vmin = 0
    vmax = 1.0

    fig, ax = plt.subplots(2, 3, figsize=(8, 4), sharex=True, sharey=True)
    ax[0, 0].imshow(np.linalg.norm(tf, axis=2), vmin=vmin, vmax=vmax)
    ax[0, 0].set_title("True flow")

    ax[1, 0].axis("off")

    ax[0, 1].imshow(dual_flow_norm, vmin=vmin, vmax=vmax)
    ax[0, 1].set_title("dualtvl10")

    ax[0, 2].imshow(farneback_flow_norm, vmin=vmin, vmax=vmax)
    ax[0, 2].set_title("farneback")

    ax[1, 1].imshow(lk_flow_norm, vmin=vmin, vmax=vmax)
    ax[1, 1].set_title("lucas kanade")

    im = ax[1, 2].imshow(bm_flow_norm, vmin=vmin, vmax=vmax)
    ax[1, 2].set_title("block matching")

    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), orientation="horizontal")
    cbar.set_label("Pixel displacement")

    vmin = 0
    vmax = 255

    fig, ax = plt.subplots(2, 3, figsize=(8, 4), sharex=True, sharey=True)
    ax[0, 0].imshow(flowiz.convert_from_flow(tf), vmin=vmin, vmax=vmax)
    ax[0, 0].set_title("True flow")

    ax[1, 0].axis("off")

    ax[0, 1].imshow(flowiz.convert_from_flow(dual_flow), vmin=vmin, vmax=vmax)
    ax[0, 1].set_title("dualtvl10")

    ax[0, 2].imshow(flowiz.convert_from_flow(farneback_flow), vmin=vmin, vmax=vmax)
    ax[0, 2].set_title("farneback")

    ax[1, 1].imshow(flowiz.convert_from_flow(lk_flow), vmin=vmin, vmax=vmax)
    ax[1, 1].set_title("lucas kanade")

    im = ax[1, 2].imshow(flowiz.convert_from_flow(bm_flow), vmin=vmin, vmax=vmax)
    ax[1, 2].set_title("block matching")

    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), orientation="horizontal")
    cbar.set_label("Pixel displacement")

    fig, ax = plt.subplots(5, 2, figsize=(6, 10), sharex=True, sharey=True)
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

    uv = flowiz.convert_from_flow(bm_flow, mode="UV")
    ax[4, 0].set_title("Horizontal Flow (U) - block matching")
    ax[4, 0].imshow(uv[..., 0], cmap=plt.get_cmap("binary"), vmin=vmin, vmax=vmax)
    ax[4, 1].set_title("Vertical Flow (V) - block matching")
    im = ax[4, 1].imshow(uv[..., 1], cmap=plt.get_cmap("binary"), vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(im, ax=ax.ravel().tolist())
    cbar.set_label("Pixel displacement")

    vmin = -5
    vmax = 5
    fig, ax = plt.subplots(5, 2, figsize=(6, 10), sharex=True, sharey=True)
    uv = tf
    ax[0, 0].set_title("Horizontal Flow (U) - True flow")
    ax[0, 0].imshow(uv[..., 0], cmap=plt.get_cmap("binary"), vmin=vmin, vmax=vmax)
    ax[0, 1].set_title("Vertical Flow (V) - True flow")
    ax[0, 1].imshow(uv[..., 1], cmap=plt.get_cmap("binary"), vmin=vmin, vmax=vmax)

    uv = dual_flow
    ax[1, 0].set_title("Horizontal Flow (U) - dualtv10")
    ax[1, 0].imshow(uv[..., 0], cmap=plt.get_cmap("binary"), vmin=vmin, vmax=vmax)
    ax[1, 1].set_title("Vertical Flow (V) - dualtvl10")
    ax[1, 1].imshow(uv[..., 1], cmap=plt.get_cmap("binary"), vmin=vmin, vmax=vmax)

    uv = farneback_flow
    ax[2, 0].set_title("Horizontal Flow (U) - farenback")
    ax[2, 0].imshow(uv[..., 0], cmap=plt.get_cmap("binary"), vmin=vmin, vmax=vmax)
    ax[2, 1].set_title("Vertical Flow (V) - farenback")
    ax[2, 1].imshow(uv[..., 1], cmap=plt.get_cmap("binary"), vmin=vmin, vmax=vmax)

    uv = lk_flow
    ax[3, 0].set_title("Horizontal Flow (U) - lucas kanade")
    ax[3, 0].imshow(uv[..., 0], cmap=plt.get_cmap("binary"), vmin=vmin, vmax=vmax)
    ax[3, 1].set_title("Vertical Flow (V) - lucas kanade")
    ax[3, 1].imshow(uv[..., 1], cmap=plt.get_cmap("binary"), vmin=vmin, vmax=vmax)

    uv = bm_flow
    ax[4, 0].set_title("Horizontal Flow (U) - block matching")
    ax[4, 0].imshow(uv[..., 0], cmap=plt.get_cmap("binary"), vmin=vmin, vmax=vmax)
    ax[4, 1].set_title("Vertical Flow (V) - block matching")
    im = ax[4, 1].imshow(uv[..., 1], cmap=plt.get_cmap("binary"), vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(im, ax=ax.ravel().tolist())
    cbar.set_label("Pixel displacement")
    # fig.tight_layout()
    plt.show()

    # flow = read_flow(flowfile.as_posix())
    # np.save(folder.joinpath("flow.npy"), flow)
    # np.save(folder.joinpath("data.npy"), np.array(frames))

    # from IPython import embed

    # embed()
    # exit()


if __name__ == "__main__":
    # dimetrodon()
    rubber_whale()
