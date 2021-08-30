from pathlib import Path

import cv2
import numpy as np
import tqdm

from . import utils


def quiver_video(
    data: utils.MPSData, vectors: np.ndarray, path: utils.PathLike
) -> None:

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    width = data.size_y
    height = data.size_x
    fps = data.framerate

    p = Path(path).with_suffix(".mp4")

    out = cv2.VideoWriter(p.as_posix(), fourcc, fps, (width, height))

    for i in tqdm.tqdm(range(data.num_frames), desc=f"Create quiver video at {p}"):
        im = utils.to_uint8(data.frames[:, :, i])
        flow = vectors[:, :, :, i]
        out.write(utils.draw_flow(im, flow))

    out.release()
    cv2.destroyAllWindows()


def hsv_video(data: utils.MPSData, vectors: np.ndarray, path: utils.PathLike) -> None:

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    width = data.size_y
    height = data.size_x
    fps = data.framerate

    p = Path(path).with_suffix(".mp4")

    out = cv2.VideoWriter(p.as_posix(), fourcc, fps, (width, height))

    for i in tqdm.tqdm(range(data.num_frames), desc=f"Create HSV movie at {p}"):
        flow = vectors[:, :, :, i]
        out.write(utils.draw_hsv(flow))

    out.release()
    cv2.destroyAllWindows()
