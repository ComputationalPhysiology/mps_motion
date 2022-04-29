import numpy as np


def to_image(A):
    # If all are equal
    B = np.array(A).astype(float)
    if ~np.any(B - B.max()):
        return B.astype(np.uint16)
    return (B / B.max()).astype(np.uint16) * (2**16 - 1)


def add_noise(A, noise_level):

    return to_image(A) + to_image(noise_level * np.random.rand(*np.shape(A)))


def create_circle_data(
    x_start=40,
    dx=5,
    y_start=40,
    dy=5,
    r=10,
    line_x=None,
    line_y=None,
    Nx=100,
    Ny=100,
):

    if line_x is None:
        line_x = np.sin(np.linspace(0, np.pi, 100))
    if line_y is None:
        line_y = np.sin(np.linspace(0, np.pi, 100))

    X = x_start + np.multiply(dx, line_x)
    Y = y_start + np.multiply(dy, line_y)

    A = []
    for x, y in zip(X, Y):
        a = np.fromfunction(
            lambda i, j: np.sqrt((i - x) ** 2 + (j - y) ** 2) < r,
            (Nx, Ny),
        )
        A.append(a)

    return to_image(A).T
