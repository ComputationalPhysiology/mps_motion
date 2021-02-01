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

from mps_motion_tracking import (
    block_matching,
    dualtvl10,
    farneback,
    lucas_kanade,
    utils,
)

# def flow2RGB(flow, max_flow_mag = 5):
#     """ Color-coded visualization of optical flow fields
#         # Arguments
#             flow: array of shape [:,:,2] containing optical flow
#             max_flow_mag: maximal expected flow magnitude used to normalize. If max_flow_mag < 0 the maximal
#             magnitude of the optical flow field will be used
#     """
#     hsv_mat = np.ones(shape=(flow.shape[0], flow.shape[1], 3), dtype=np.float32) * 255
#     ee = cv2.sqrt(flow[:, :, 0] * flow[:, :, 0] + flow[:, :, 1] * flow[:, :, 1])
#     angle = np.arccos(flow[:, :, 0]/ ee)
#     angle[flow[:, :, 0] == 0] = 0
#     angle[flow[:, :, 1] == 0] = 6.2831853 - angle[flow[:, :, 1] == 0]
#     angle = angle * 180 / 3.141
#     hsv_mat[:,:,0] = angle
#     if max_flow_mag < 0:
#         max_flow_mag = ee.max()
#     hsv_mat[:,:,1] = ee * 255.0 / max_flow_mag
#     ret, hsv_mat[:,:,1] = cv2.threshold(src=hsv_mat[:,:,1], maxval=255, thresh=255, type=cv2.THRESH_TRUNC )
#     rgb_mat = cv2.cvtColor(hsv_mat.astype(np.uint8), cv2.COLOR_HSV2BGR)
#     return rgb_mat


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

    dual_flow = dualtvl10.flow(frames[1], frames[0])
    dual_flow_norm = np.linalg.norm(dual_flow, axis=2)
    dual_flow_norm /= dual_flow_norm.max()

    farneback_flow = farneback.flow(
        frames[1],
        frames[0],
    )
    farneback_flow_norm = np.linalg.norm(farneback_flow, axis=2)
    farneback_flow_norm /= farneback_flow_norm.max()

    points = lucas_kanade.get_uniform_reference_points(frames[0], step=4)

    lk_flow_ = lucas_kanade.flow(frames[1], frames[0], points)
    lk_flow = utils.rbfinterp2d(
        points.squeeze(),
        lk_flow_,
        np.arange(frames[0].shape[1]),
        np.arange(frames[0].shape[0]),
    ).T

    lk_norm_flow = np.linalg.norm(lk_flow, axis=0).T
    lk_norm_flow /= lk_norm_flow.max()

    # from IPython import embed

    # embed()
    # exit()
    bm_flow = block_matching.filter_vectors(
        block_matching.flow(frames[0], frames[1]), 5
    )
    bm_norm = np.linalg.norm(bm_flow, axis=2)
    bm_norm_flow = cv2.resize(bm_norm, tuple(reversed(frames[0].shape)))
    bm_norm_flow /= bm_norm_flow.max()

    vmin = 0
    vmax = 1.0

    fig, ax = plt.subplots(2, 3)  # , sharex=True, sharey=True)
    ax[0, 0].imshow(np.linalg.norm(tf, axis=2), vmin=vmin, vmax=vmax)
    ax[0, 0].set_title("True flow")

    ax[0, 1].imshow(dual_flow_norm, vmin=vmin, vmax=vmax)
    ax[0, 1].set_title("dualtvl10")

    ax[1, 0].imshow(farneback_flow_norm, vmin=vmin, vmax=vmax)
    ax[1, 0].set_title("farneback")

    ax[1, 1].imshow(lk_norm_flow, vmin=vmin, vmax=vmax)
    ax[1, 1].set_title("lucas kanade")

    ax[1, 2].imshow(bm_norm_flow, vmin=vmin, vmax=vmax)
    ax[1, 2].set_title("block matching")
    plt.show()
    exit()
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
    # dimetrodon()
    rubber_whale()
