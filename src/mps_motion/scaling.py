import logging
from typing import Optional
from typing import Tuple

import cv2
import dask
import dask.array as da
import numpy as np
import scipy.spatial
import tqdm
from dask.diagnostics import ProgressBar

from . import utils

logger = logging.getLogger(__name__)

INTERPOLATION_METHODS = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


def resize_vectors(vectors, new_shape):
    if len(vectors.shape) == 4:
        vec1 = resize_frames(
            vectors[:, :, :, 0],
            new_shape=new_shape,
        )
        vec2 = resize_frames(
            vectors[:, :, :, 1],
            new_shape=new_shape,
        )
    else:
        assert len(vectors.shape) == 3
        vec1 = resize_frames(
            vectors[:, :, 0],
            new_shape=new_shape,
        )
        vec2 = resize_frames(
            vectors[:, :, 1],
            new_shape=new_shape,
        )
    return da.stack([vec1, vec2], axis=-1)


def resize_data(data: utils.MPSData, scale: float) -> utils.MPSData:
    new_frames = resize_frames(data.frames, scale)
    info = data.info.copy()
    info["um_per_pixel"] /= scale
    info["size_x"], info["size_y"], info["num_frames"] = new_frames.shape
    return utils.MPSData(new_frames, data.time_stamps, info)


def subsample_time(data: utils.MPSData, step: int) -> utils.MPSData:

    new_frames = data.frames[:, :, ::step]
    new_times = data.time_stamps[::step]
    info = data.info.copy()
    info["num_frames"] = len(new_times)

    return utils.MPSData(new_frames, new_times, info)


def reshape_lk(reference_points: np.ndarray, flows: utils.Array) -> utils.Array:
    x, y = reference_points.reshape(-1, 2).astype(int).T
    xu = np.sort(np.unique(x))
    yu = np.sort(np.unique(y))

    is_dask = False
    if isinstance(flows, da.Array):
        is_dask = True

    dx = xu[0]
    dxs = np.diff(xu)
    if len(dxs) > 0:
        dx = dxs[0]
        assert np.all(dxs == dx)

    dy = yu[0]
    dys = np.diff(yu)
    if len(dys) > 0:
        dy = dys[0]
        assert np.all(dys == dy)

    xp = ((x - np.min(x)) / dx).astype(int)
    yp = ((y - np.min(y)) / dy).astype(int)
    if len(flows.shape) == 3:
        num_frames = flows.shape[-1]
        out = np.zeros((yp.max() + 1, xp.max() + 1, 2, num_frames))
        out[yp, xp, :, :] = flows
        out = np.swapaxes(out, 2, 3)
    else:
        out = np.zeros((yp.max() + 1, xp.max() + 1, 2))
        out[yp, xp, :] = flows

    if is_dask:
        out = da.from_array(out)
    return out


def resize_frames(
    frames: np.ndarray,
    scale: float = 1.0,
    new_shape: Optional[Tuple[int, int]] = None,
    interpolation_method="nearest",
) -> np.ndarray:
    logger.info("Resize frames")
    msg = f"Expected interpolation method to be one of {INTERPOLATION_METHODS.keys()}, got {interpolation_method}"
    assert interpolation_method in INTERPOLATION_METHODS, msg
    if scale != 1.0 or new_shape is not None:

        if len(frames.shape) == 2:
            w, h = frames.shape
        else:
            w, h, num_frames = frames.shape

        if new_shape is not None:
            assert len(new_shape) == 2
            width, height = new_shape
        else:
            width = int(w * scale)
            height = int(h * scale)

        width = int(width)
        height = int(height)

        if len(frames.shape) == 2:
            return cv2.resize(frames, (height, width))

        all_resized_frames = []
        for i in range(num_frames):
            all_resized_frames.append(
                dask.delayed(cv2.resize)(
                    frames[:, :, i],
                    (height, width),
                    INTERPOLATION_METHODS[interpolation_method],
                ),
            )
        with ProgressBar(out=utils.LoggerWrapper(logger, "info")):
            resized_frames = da.stack(
                *da.compute(all_resized_frames), axis=-1
            ).compute()

    else:
        resized_frames = frames.copy()
    logger.info("Done resizing")
    return resized_frames


def interpolate_lk_flow(
    disp: np.ndarray,
    reference_points: np.ndarray,
    size_x: int,
    size_y: int,
    interpolation_method: str = "linear",
) -> np.ndarray:
    """Given an array of displacements (of flow) coming from
    the Lucas Kanade method return a new array which
    interpolates the data onto a given size, i.e
    the original size of the image.

    Parameters
    ----------
    disp : np.ndarray
        The flow or displacement from LK algorithm
    reference_points : np.ndarray
        Reference points
    size_x : int
        Size of the output in x-direction
    size_y : int
        Size of the output in y-direction
    interpolation_method : str
        Method for interpolation, by default 'linear'

    Returns
    -------
    np.ndarray
        Interpolated values
    """
    num_frames = disp.shape[-1]
    from scipy.interpolate import griddata

    disp_full = np.zeros((size_y, size_x, 2, num_frames))
    ref_points = np.squeeze(reference_points)
    grid_x, grid_y = np.meshgrid(np.arange(size_x), np.arange(size_y))
    # TODO: This could be parallelized
    for i in tqdm.tqdm(range(num_frames)):
        values_x = disp[:, 0, i]
        values_y = disp[:, 1, i]

        disp_full[:, :, 0, i] = griddata(
            ref_points,
            values_x,
            (grid_y, grid_x),
            method=interpolation_method,
        )
        disp_full[:, :, 1, i] = griddata(
            ref_points,
            values_y,
            (grid_y, grid_x),
            method=interpolation_method,
        )
    return disp_full


def rbfinterp2d_map(args):
    return rbfinterp2d(*args)


def rbfinterp2d(  # noqa:C901
    coord,
    input_array,
    xgrid,
    ygrid,
    rbfunction="gaussian",
    epsilon=10,
    k=50,
    nchunks=5,
):
    """
    Fast 2-D grid interpolation of a sparse (multivariate) array using a
    radial basis function.

    https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html

    Parameters
    ----------
    coord: array_like
        Array of shape (n, 2) containing the coordinates of the data points
        into a 2-dimensional space.
    input_array: array_like
        Array of shape (n) or (n, m) containing the values of the data points,
        where *n* is the number of data points and *m* the number of co-located
        variables. All values in ``input_array`` are required to have finite values.
    xgrid, ygrid: array_like
        1D arrays representing the coordinates of the 2-D output grid.
    rbfunction: {"gaussian", "multiquadric", "inverse quadratic", "inverse multiquadric", "bump"}, optional
        The name of one of the available radial basis function based on a
        normalized Euclidian norm as defined in the **Notes** section below.
        More details provided in the wikipedia reference page linked below.
    epsilon: float, optional
        The shape parameter used to scale the input to the radial kernel.
        A smaller value for ``epsilon`` produces a smoother interpolation. More
        details provided in the wikipedia reference page linked below.
    k: int or None, optional
        The number of nearest neighbours used for each target location.
        This can also be useful to to speed-up the interpolation.
        If set to None, it interpolates using all the data points at once.
    nchunks: int, optional
        The number of chunks in which the grid points are split to limit the
        memory usage during the interpolation.

    Returns
    -------
    output_array: ndarray
        The interpolated field(s) having shape (*m*, ``ygrid.size``, ``xgrid.size``).

    Notes
    -----
    The coordinates are normalized before computing the Euclidean norms:

    .. math::
        x = (x - min(x)) / max[max(x) - min(x), max(y) - min(y)],
        y = (y - min(y)) / max[max(x) - min(x), max(y) - min(y)],

    where the min and max values are taken as the 2nd and 98th percentiles.

    References
    ----------
    Wikipedia contributors, "Radial basis function,"
    Wikipedia, The Free Encyclopedia,
    https://en.wikipedia.org/w/index.php?title=Radial_basis_function&oldid=906155047
    (accessed August 19, 2019).

    """

    _rbfunctions = [
        "nearest",
        "gaussian",
        "inverse quadratic",
        "inverse multiquadric",
        "bump",
    ]

    input_array = np.copy(input_array)

    if np.any(~np.isfinite(input_array)):
        raise ValueError("input_array contains non-finite values")

    if input_array.ndim == 1:
        nvar = 1
        input_array = input_array[:, None]

    elif input_array.ndim == 2:
        nvar = input_array.shape[1]

    else:
        raise ValueError(
            "input_array must have 1 (n) or 2 dimensions (n, m), but it has %i"
            % input_array.ndim,
        )

    npoints = input_array.shape[0]

    if npoints == 0:
        raise ValueError(
            "input_array (n, m) must contain at least one sample, but it has %i"
            % npoints,
        )

    # only one sample, return uniform fields
    elif npoints == 1:
        output_array = np.ones((nvar, ygrid.size, xgrid.size))
        for i in range(nvar):
            output_array[i, :, :] *= input_array[:, i]
        return output_array

    coord = np.copy(coord)

    if coord.ndim != 2:
        raise ValueError(
            "coord must have 2 dimensions (n, 2), but it has %i" % coord.ndim,
        )

    if npoints != coord.shape[0]:
        raise ValueError(
            "the number of samples in the input_array does not match the "
            + "number of coordinates %i!=%i" % (npoints, coord.shape[0]),
        )

    # normalize coordinates
    qcoord = np.percentile(coord, [2, 98], axis=0)
    dextent = np.max(np.diff(qcoord, axis=0))
    coord = (coord - qcoord[0, :]) / dextent

    rbfunction = rbfunction.lower()
    if rbfunction not in _rbfunctions:
        raise ValueError(
            "Unknown rbfunction '{}'\n".format(rbfunction)
            + "The available rbfunctions are: "
            + str(_rbfunctions),
        ) from None

    # generate the target grid
    X, Y = np.meshgrid(xgrid, ygrid)
    grid = np.column_stack((X.ravel(), Y.ravel()))
    # normalize the grid coordinates
    grid = (grid - qcoord[0, :]) / dextent

    # k-nearest interpolation
    if k is not None and k > 0:
        k = int(np.min((k, npoints)))

        # create cKDTree object to represent source grid
        tree = scipy.spatial.cKDTree(coord)

    else:
        k = 0

    # split grid points in n chunks
    if nchunks > 1:
        subgrids = np.array_split(grid, nchunks, 0)
        subgrids = [x for x in subgrids if x.size > 0]

    else:
        subgrids = [grid]

    # loop subgrids
    i0 = 0
    output_array = np.zeros((grid.shape[0], nvar))
    for i, subgrid in enumerate(subgrids):
        idelta = subgrid.shape[0]

        if k == 0:
            # use all points
            d = scipy.spatial.distance.cdist(coord, subgrid, "euclidean").transpose()
            inds = np.arange(npoints)[None, :] * np.ones(
                (subgrid.shape[0], npoints),
            ).astype(int)

        else:
            # use k-nearest neighbours
            d, inds = tree.query(subgrid, k=k)

        if k == 1:
            # nearest neighbour
            output_array[i0 : (i0 + idelta), :] = input_array[inds, :]

        else:

            # the interpolation weights
            if rbfunction == "gaussian":
                w = np.exp(-((d * epsilon) ** 2))

            elif rbfunction == "inverse quadratic":
                w = 1.0 / (1 + (epsilon * d) ** 2)

            elif rbfunction == "inverse multiquadric":
                w = 1.0 / np.sqrt(1 + (epsilon * d) ** 2)

            elif rbfunction == "bump":
                w = np.exp(-1.0 / (1 - (epsilon * d) ** 2))
                w[d >= 1 / epsilon] = 0.0

            if not np.all(np.sum(w, axis=1)):
                w[np.sum(w, axis=1) == 0, :] = 1.0

            # interpolate
            for j in range(nvar):
                output_array[i0 : (i0 + idelta), j] = np.sum(
                    w * input_array[inds, j],
                    axis=1,
                ) / np.sum(w, axis=1)

        i0 += idelta

    # reshape to final grid size
    return output_array.reshape(ygrid.size, xgrid.size, nvar)
