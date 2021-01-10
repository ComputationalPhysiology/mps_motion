import cv2
import mps
import numpy as np
import tqdm

from mps_motion_tracking import lucas_kanade, utils


def main():

    data = mps.MPS("PointH4A_ChannelBF_VC_Seq0018.nd2")
    disp, ref_points = lucas_kanade.get_displacements(data.frames, data.frames[:, :, 0])
    np.save("lk_disp.npy", disp)
    np.save("lk_ref_points.npy", ref_points)


def postprocess_displacement():
    data = mps.MPS("PointH4A_ChannelBF_VC_Seq0018.nd2")
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


if __name__ == "__main__":
    # main()
    postprocess_displacement()
