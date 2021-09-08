from pathlib import Path

import cv2
import numpy as np
import tqdm

from . import utils


def _draw_flow(image, x, y, fx, fy):
    QUIVER = (0, 0, 255)
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, QUIVER, thickness=5, lineType=8)
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_lk_flow(image, flow, reference_points):
    x, y = reference_points.reshape(-1, 2).astype(int).T
    fx, fy = flow.T
    return _draw_flow(image, x, y, fx, fy)


def draw_flow(image, flow, step=16, scale: float = 1.0):
    """[summary]

    Parameters
    ----------
    image : [type]
        [description]
    flow : [type]
        [description]
    step : int, optional
        [description], by default 16

    Returns
    -------
    [type]
        [description]
    """
    h, w = image.shape[:2]
    y, x = np.mgrid[step / 2 : h : step, step / 2 : w : step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    return _draw_flow(image, x, y, scale * fx, scale * fy)


def draw_hsv(flow):
    h, w = flow.shape[:2]
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((h, w, 3), np.uint8)
    # Sets image hue according to the optical flow
    # direction
    hsv[..., 0] = angle * 180 / np.pi / 2

    # Sets image saturation to maximum
    hsv[..., 1] = 255

    # Sets image value according to the optical flow
    # magnitude (normalized)
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return rgb


def quiver_video(
    data: utils.MPSData,
    vectors: np.ndarray,
    path: utils.PathLike,
    step: int = 16,
    scale: float = 1.0,
    convert: bool = True,
    velocity: bool = False,
) -> None:

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    width = data.size_y
    height = data.size_x
    fps = data.framerate

    p = Path(path).with_suffix(".mp4")
    if p.is_file():
        p.unlink()
    out = cv2.VideoWriter(p.as_posix(), fourcc, fps, (width, height))

    num_frames = data.num_frames - 1 if velocity else data.num_frames
    time_axis = vectors.shape.index(num_frames)
    for i in tqdm.tqdm(range(num_frames), desc=f"Create quiver video at {p}"):
        im = utils.to_uint8(data.frames[:, :, i])
        flow = np.take(vectors, i, axis=time_axis)
        out.write(draw_flow(im, flow, step=step, scale=scale))

    out.release()
    cv2.destroyAllWindows()

    if convert:
        import imageio

        tmp_path = p.parent.joinpath(p.stem + "_tmp").with_suffix(".mp4")
        p.rename(tmp_path)
        video = imageio.read(tmp_path)
        imageio.mimwrite(p, (np.swapaxes(d, 0, 1) for d in video.iter_data()), fps=fps)
        tmp_path.unlink()


def hsv_video(
    data: utils.MPSData,
    vectors: np.ndarray,
    path: utils.PathLike,
    convert: bool = False,
) -> None:

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    width = data.size_y
    height = data.size_x
    fps = data.framerate

    p = Path(path).with_suffix(".mp4")

    out = cv2.VideoWriter(p.as_posix(), fourcc, fps, (width, height))
    time_axis = vectors.shape.index(data.num_frames)
    for i in tqdm.tqdm(range(data.num_frames), desc=f"Create HSV movie at {p}"):
        flow = np.take(vectors, i, axis=time_axis)
        out.write(draw_hsv(flow))

    out.release()
    cv2.destroyAllWindows()

    if convert:
        import imageio

        tmp_path = p.parent.joinpath(p.stem + "_tmp").with_suffix(".mp4")
        p.rename(tmp_path)
        video = imageio.read(tmp_path)
        imageio.mimwrite(p, video.iter_data(), fps=fps)
