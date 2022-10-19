try:
    import streamlit as st
except ImportError:
    print("Please install streamlit - python3 -m pip install streamlit")
    exit(1)

from pathlib import Path
import matplotlib.pyplot as plt


import tempfile
import sys
import mps
import mps_motion


def get_folder_and_files():
    folder = Path(sys.argv[1])
    return folder, [
        f.name
        for f in folder.iterdir()
        if f.suffix in [".nd2", ".tif", ".tiff", ".czi"]
    ]


def video():
    st.title("Video")

    folder, files = get_folder_and_files()
    filename = st.selectbox("Select file", options=files)

    if st.button("Show video"):
        with st.spinner("Loading video"):
            data = mps.MPS(folder / filename)
            with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
                mps.utils.frames2mp4(
                    data.frames.T,
                    path=f.name,
                    framerate=data.framerate,
                )
                st.video(open(f.name, "rb"), format="video/mp4")


def motion_analysis():
    st.title("Motion analysis")

    folder, files = get_folder_and_files()

    filename = st.selectbox("Select file", options=files)

    cols_mech = st.columns(2)
    with cols_mech[0]:
        flow_algorithm = st.selectbox(
            "Optical flow algorithm",
            mps_motion.list_optical_flow_algorithm(),
        )
        reference_frame = st.number_input("Reference frame", 0)
        show_motion_video = st.checkbox("Show video")

        if show_motion_video:
            vector_scale = st.number_input("Vector scale", value=1, min_value=1)
            step = st.number_input("Step", value=16, min_value=1)

    with cols_mech[1]:
        spacing = st.number_input("spacing", value=1, min_value=1)
        scale = st.number_input("Scale", value=0.4, min_value=0.1, max_value=1.0)

    if not st.button("Run motion analysis"):
        return

    print("Run motion analysis")
    with st.spinner("Read data"):
        data = mps.MPS(folder / filename)

    opt_flow = mps_motion.OpticalFlow(
        data,
        flow_algorithm=flow_algorithm,
        reference_frame=reference_frame,
    )

    with st.spinner("Compute displacement"):
        u = opt_flow.get_displacements(scale=scale)

    mech = mps_motion.Mechanics(u=u, t=data.time_stamps)

    with st.spinner("Compute velocity"):
        v = mech.velocity(spacing=spacing)

    with st.spinner("Compute mean norm displacement"):
        u_norm_mean = u.norm().mean().compute()

    with st.spinner("Compute mean norm velocity"):
        v_norm_mean = v.norm().mean()

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(data.time_stamps, u_norm_mean)
    ax[0].set_ylabel("Displacement [\u00B5m]")
    ax[1].plot(data.time_stamps[:-spacing], v_norm_mean)
    ax[1].set_ylabel("Velocity [\u00B5m/s]")
    st.pyplot(fig=fig)

    if show_motion_video:
        with st.spinner("Creating video"):
            scaled_data = data
            if scale < 1.0:
                scaled_data = mps_motion.scaling.resize_data(data, scale=u.scale)

            with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
                mps_motion.visu.quiver_video(
                    scaled_data,
                    u,
                    f.name,
                    step=step,
                    vector_scale=vector_scale,
                )
                st.video(open(f.name, "rb"), format="video/mp4")

    return


if __name__ == "__main__":
    # Page settings
    st.set_page_config(page_title="MPS motion gui")

    # Sidebar settings
    pages = {
        "Motion analysis": motion_analysis,
        "Video": video,
    }

    st.sidebar.title("mps-motion")

    # Radio buttons to select desired option
    page = st.sidebar.radio("Pages", tuple(pages.keys()))

    pages[page]()

    # About
    st.sidebar.markdown(
        """
    - [Source code](https://github.com/ComputationalPhysiology/mps_motion)
    - [Documentation](http://computationalphysiology.github.io/mps_motion)
    """,
    )
