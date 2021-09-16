import cv2
import matplotlib.pyplot as plt
import mps
import numpy as np
import tqdm

from mps_motion_tracking import farneback
from mps_motion_tracking import utils


def load_test_data():
    return mps.MPS.from_dict(**np.load("mps_data.npy", allow_pickle=True).item())


def main():
    data = load_test_data()
    # ref = np.mean(data.frames, axis=2)
    # ref = np.min(data.frames, axis=2)
    # ref = np.max(data.frames, axis=2)
    # ref = np.median(data.frames, axis=2)
    ref = data.frames[:, :, 0]
    disp = farneback.get_displacements(data.frames, ref) * data.info["um_per_pixel"]

    q = np.linalg.norm(disp, axis=2).mean((0, 1))

    plt.plot(data.time_stamps, q)
    plt.show()

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


def resize_test_data():

    data = mps.MPS("../sandbox/opencv_optical_flow/PointH4A_ChannelBF_VC_Seq0018.nd2")

    scale = 0.1
    new_frames = utils.resize_frames(data.frames[:, :, 40:100], scale)
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(data.frames[:, :, 0])
    # ax[1].imshow(new_frames[:, :, 0])
    # plt.show()

    info = data.info.copy()
    info["um_per_pixel"] /= scale
    info["size_x"], info["size_y"], info["num_frames"] = new_frames.shape

    np.save(
        "mps_data.npy",
        {"frames": new_frames, "time_stamps": data.time_stamps[:60], "info": info},
    )


if __name__ == "__main__":
    resize_test_data()
    main()
