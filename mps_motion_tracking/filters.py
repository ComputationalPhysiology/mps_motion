import logging
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from typing import List
from typing import Optional

import dask
import dask.array as da
import dask_image.ndfilters
import numpy as np
import tqdm
from dask.diagnostics import ProgressBar
from scipy.interpolate import bisplev
from scipy.interpolate import bisplrep
from typing_extensions import Protocol

from . import utils

logger = logging.getLogger(__name__)


def spline_smooth_u(args):
    ux, uy, X, Y = args

    N = min(ux.shape) // 10

    x = X.flatten()[::N]
    y = Y.flatten()[::N]
    valuesx = ux.flatten()[::N]
    valuesy = uy.flatten()[::N]

    sx = X.shape[1] - np.sqrt(2 * X.shape[1])
    sy = X.shape[0] - np.sqrt(2 * X.shape[0])

    tckx = bisplrep(x, y, valuesx, kx=5, ky=5, s=sx)
    Ux = bisplev(X[:, 0], Y[0, :], tckx)
    tcky = bisplrep(x, y, valuesy, kx=5, ky=5, s=sy)
    Uy = bisplev(X[:, 0], Y[0, :], tcky)

    return [Ux, Uy]


def process_spline_interpolation(u):
    x = np.arange(u.shape[0])
    y = np.arange(u.shape[1])
    Y, X = np.meshgrid(y, x)

    Us = []

    args = ((u[:, :, i, 0], u[:, :, i, 1], X, Y) for i in range(u.shape[2]))

    with ProcessPoolExecutor() as executor:
        for ui in tqdm.tqdm(
            executor.map(spline_smooth_u, args),
            desc="Running spline interpolation",
            total=u.shape[2],
        ):
            Us.append(ui)

    return Us


def spline_smooth(u: utils.Array) -> np.ndarray:

    if isinstance(u, da.Array):
        u = u.compute()

    logger.info("Performing spline interpolation, this may take some time...")
    Us = process_spline_interpolation(u)

    new_u = np.zeros(u.shape)

    for i, ui in enumerate(Us):
        new_u[:, :, i, 0] = ui[0]
        new_u[:, :, i, 1] = ui[1]

    return new_u


class Filters(str, Enum):
    median = "median"
    gaussian = "gaussian"


def filter_vectors(
    vectors: utils.Array,
    filter_type: Filters,
    size: Optional[int] = None,
    sigma: Optional[float] = None,
):

    if not valid_filter(filter_type=filter_type, size=size, sigma=sigma):
        return vectors

    is_numpy = False
    if isinstance(vectors, np.ndarray):
        is_numpy = True
        vectors = da.from_array(vectors)

    vec0 = apply_filter(
        vectors[:, :, 0],
        filter_type=filter_type,
        size=size,
        sigma=sigma,
    )
    vec1 = apply_filter(
        vectors[:, :, 1],
        filter_type=filter_type,
        size=size,
        sigma=sigma,
    )
    vectors = da.stack([vec0, vec1], axis=-1)
    if is_numpy:
        vectors = vectors.compute()
    return vectors


def filter_vectors_par(
    vectors,
    filter_type: Filters,
    size: Optional[int] = None,
    sigma: Optional[float] = None,
):

    if not valid_filter(filter_type=filter_type, size=size, sigma=sigma):
        return vectors
    logger.info("Filter vectors")
    assert len(vectors.shape) == 4
    assert vectors.shape[3] == 2
    num_frames = vectors.shape[2]

    is_numpy = False
    if isinstance(vectors, np.ndarray):
        is_numpy = True
        vectors = da.from_array(vectors)

    all_vectors = []
    for i in range(num_frames):
        all_vectors.append(
            dask.delayed(filter_vectors)(
                vectors[:, :, i, :],
                filter_type=filter_type,
                size=size,
                sigma=sigma,
            ),
        )

    with ProgressBar():
        vectors = da.stack(*da.compute(all_vectors), axis=2)
    if is_numpy:
        vectors = vectors.compute()

    logger.info("Done filtering")
    return vectors


def valid_filter(
    filter_type: Filters,
    size: Optional[int] = None,
    sigma: Optional[float] = None,
) -> bool:
    if filter_type == Filters.median and size is None:
        logger.warning("Please provide a size of the median filter kernel")
        return False
    elif filter_type == Filters.gaussian and sigma is None:
        logger.warning("Please provide a sigma for the gaussian filter")
        return False
    if filter_type not in Filters._member_names_:
        logger.warning(f"Unknown filter type {filter_type}")
        return False
    return True


def is_positive(value: Optional[float]) -> bool:
    if value is None:
        return False
    return value > 0


def apply_filter(
    array: utils.Array,
    filter_type: Filters,
    size: Optional[int] = None,
    sigma: Optional[float] = None,
) -> utils.Array:
    if not valid_filter(filter_type=filter_type, size=size, sigma=sigma):
        return array

    is_numpy = False
    if isinstance(array, np.ndarray):
        is_numpy = True
        array = da.from_array(array)

    if filter_type == Filters.median and is_positive(size):
        array = dask_image.ndfilters.median_filter(array, size)

    elif filter_type == Filters.gaussian and is_positive(sigma):
        array = dask_image.ndfilters.gaussian_filter(array, sigma)

    if is_numpy:
        array = array.compute()

    return array


class InvalidThresholdError(ValueError):
    pass


def check_threshold(
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    if (vmax and vmin) and vmax < vmin:
        raise InvalidThresholdError(f"Cannot have vmax < vmin, got {vmax=} and {vmin=}")


def threshold(
    array: utils.Array,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    copy: bool = True,
) -> utils.Array:
    """Threshold an array

    Parameters
    ----------
    array : utils.Array
        The array
    vmin : Optional[float], optional
        Lower threshold value, by default None
    vmax : Optional[float], optional
        Upper threshold value, by default None
    copy : bool, optional
        Operative on the given array or use a copy, by default True

    Returns
    -------
    utils.Array
        Inpute array with the lowest value beeing vmin and
        highest value begin vmax
    """
    assert len(array.shape) == 3
    check_threshold(vmin, vmax)
    if copy:
        array = array.copy()

    if vmax is not None:
        array[array > vmax] = vmax
    if vmin is not None:
        array[array < vmin] = vmin
    return array


def _handle_threshold_norm(norm_inds, ns, factor, norm_array, array):
    if norm_inds.any():
        if ns == da:
            norm_inds = norm_inds.compute()
        inds = np.stack([norm_inds, norm_inds], -1).flatten()
        values = (
            factor
            / ns.stack([norm_array[norm_inds], norm_array[norm_inds]], -1).flatten()
        )
        array[inds] *= values


class _Linalg(Protocol):
    @staticmethod
    def norm(array: utils.Array, axis: int = 0) -> utils.Array:
        ...


class NameSpace(Protocol):
    @property
    def linalg(self) -> _Linalg:
        ...

    @staticmethod
    def stack(arrs: List[utils.Array], axis: int) -> utils.Array:
        ...


def threshold_norm(
    array: utils.Array,
    ns: NameSpace,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    copy: bool = True,
) -> utils.Array:
    """Threshold an array of vectors based on the
    norm of the vectors.

    For example if the vectors are displacement then
    you can use this function to scale all vectors so
    that the magnitudes are within `vmin` and `vmax`

    Parameters
    ----------
    array : utils.Array
        The input array which is 4D and the last dimension is 2.
    ns : NameSpace
        Wheter to use numpy or dask
    vmin : Optional[float], optional
        Lower bound on the norm, by default None
    vmax : Optional[float], optional
        Upper bound on the norm, by default None
    copy : bool, optional
        Wheter to operate on the input are or use a copy, by default True

    Returns
    -------
    utils.Array
        The thresholded array
    """
    assert len(array.shape) == 4
    assert array.shape[3] == 2
    assert ns in [da, np]
    check_threshold(vmin, vmax)
    if copy:
        array = array.copy()
    shape = array.shape
    norm_array = ns.linalg.norm(array, axis=3).flatten()
    array = array.flatten()

    if vmax is not None:
        norm_inds = norm_array > vmax
        _handle_threshold_norm(norm_inds, ns, vmax, norm_array, array)
    if vmin is not None:
        norm_inds = norm_array < vmin
        _handle_threshold_norm(norm_inds, ns, vmin, norm_array, array)
    return array.reshape(shape)
