from textwrap import dedent
from typing import Optional

import typer

from .cli import main as _main
from .motion_tracking import FLOW_ALGORITHMS

app = typer.Typer(help="Estimate motion in stack of images")


@app.command()
def main(
    filename: str = typer.Argument(
        ...,
        help=dedent(
            """
        Path to file to be analyzed, typically an .nd2 or .
        czi Brightfield file
        """,
        ),
    ),
    algorithm: FLOW_ALGORITHMS = typer.Option(
        FLOW_ALGORITHMS.farneback,
        help="The algorithm used to estimate motion",
    ),
    reference_frame: str = typer.Option(
        "0",
        "--reference-frame",
        "-rf",
        help=dedent(
            """
        Which frame should be the reference frame when computing the
        displacements. This can either be a number indicating the
        timepoint, or the value 'mean', 'median', 'max' or 'mean'.
        Default: '0' (i.e the first frame)
        """,
        ),
    ),
    scale: float = typer.Option(
        1.0,
        help=dedent(
            """
        Rescale data before running motion track. This is useful if the spatial resoltion
        of the images are large. Scale = 1.0 will keep the original size
        """,
        ),
    ),
    apply_filter: bool = typer.Option(
        True,
        help=dedent(
            """
            If True, set pixels with max displacement lower than the mean maximum displacement
            to zero. This will prevent non-tissue pixels to be included, which is especially
            error prone for velocity estimations, by default True.""",
        ),
    ),
    spacing: int = typer.Option(
        5,
        help=dedent(
            """Spacing between frames in velocity computations, by default 5.
            """,
        ),
    ),
    compute_xy_components: bool = typer.Option(
        False,
        "--xy",
        "-xy",
        help=dedent(
            """
            If True the compute x- and y components of the displacement and
            velocity and plot them as well, by default False.""",
        ),
    ),
    make_displacement_video: bool = typer.Option(
        False,
        "--video-disp",
        help=dedent(
            """
            If True, create video of displacement vectors, by default False.""",
        ),
    ),
    make_velocity_video: bool = typer.Option(
        False,
        "--video-vel",
        help=dedent(
            """
            If True, create video of velocity vectors, by default False.""",
        ),
    ),
    outdir: Optional[str] = typer.Option(
        None,
        "--outdir",
        "-o",
        help=dedent(
            """
        Directory where to store the results. If not provided, a folder with the the same
        as the filename will be created and results will be stored in a subfolder inside
        that called `motion`
        """,
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="More verbose"),
):
    _main(
        filename=filename,
        algorithm=algorithm,
        reference_frame=reference_frame,
        outdir=outdir,
        scale=scale,
        apply_filter=apply_filter,
        spacing=spacing,
        compute_xy_components=compute_xy_components,
        make_displacement_video=make_displacement_video,
        make_velocity_video=make_velocity_video,
        verbose=verbose,
    )


def run():
    typer.run(main)


if __name__ == "__main__":
    run()
