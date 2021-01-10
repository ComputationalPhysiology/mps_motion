import cv2
import mps
import numpy as np
import tqdm

from mps_motion_tracking import dualtvl10, utils


def main():

    data = mps.MPS("PointH4A_ChannelBF_VC_Seq0018.nd2")
    disp = dualtvl10.get_displacements(data.frames, data.frames[:, :, 0])
    np.save("dualtvl10_disp.npy", disp)
    vel = dualtvl10.get_velocities(data.frames, data.frames[:, :, 0])
    np.save("dualtvl10_vel.npy", vel)


def postprocess_displacement():
    data = mps.MPS("PointH4A_ChannelBF_VC_Seq0018.nd2")
    disp = np.load("dualtvl10_disp.npy")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    width = data.size_y
    height = data.size_x
    fps = data.framerate

    out_flow = cv2.VideoWriter(
        "dualtvl10_displacement_flow.mp4", fourcc, fps, (width, height)
    )
    out_hsv = cv2.VideoWriter(
        "dualtvl10_displacement_hsv.mp4", fourcc, fps, (width, height)
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
    data = mps.MPS("PointH4A_ChannelBF_VC_Seq0018.nd2")
    disp = np.load("dualtvl10_vel.npy")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    width = data.size_y
    height = data.size_x
    fps = data.framerate

    out_flow = cv2.VideoWriter(
        "dualtvl10_velocity_flow.mp4", fourcc, fps, (width, height)
    )
    out_hsv = cv2.VideoWriter(
        "dualtvl10_velocity_hsv.mp4", fourcc, fps, (width, height)
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


if __name__ == "__main__":
    main()
    postprocess_displacement()
    postprocess_velocities()
