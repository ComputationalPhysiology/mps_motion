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
    normalize_baseline: bool = typer.Option(
        False,
        "--normalize-baseline",
        "-n",
        help=dedent(
            """
            If True, subtract value at time of referece frame so that
            e.g displacement at this point becomes zero.""",
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
    scale: float = typer.Option(
        0.3,
        help=dedent(
            """
        Rescale data before running motion track. This is useful if the spatial resoltion
        of the images are large. Scale = 1.0 will keep the original size
        """,
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="More verbose"),
    overwrite: bool = typer.Option(
        True,
        help=dedent(
            """
        If True, overwrite existing data if outdir
        allready exist. If False, then the olddata will
        be copied to a subdirectory with version number
        of the software. If version number is not found
        it will be saved to a folder called "old".""",
        ),
    ),
):
    _main(
        filename=filename,
        algorithm=algorithm,
        reference_frame=reference_frame,
        normalize_baseline=normalize_baseline,
        outdir=outdir,
        scale=scale,
        verbose=verbose,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    typer.run(main)
