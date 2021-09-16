#!/usr/bin/env python
"""The setup script."""
from setuptools import find_packages
from setuptools import setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

requirements = ["numpy", "opencv-python", "tqdm", "dask[array]", "typer"]

extras_require = {
    "task-queue": ["redis", "rq", "SQLAlchemy", "Flask", "Flask-SQLAlchemy"],
    "benchmark": ["flowiz"],
    "dualtvl10": ["opencv-contrib-python"],
    "block_matching": ["numba", "scipy"],
    "legacy": ["cached_property"],
    "h5py": ["h5py"],
}

extras_require.update(
    {"all": [val for values in extras_require.values() for val in values]},
)

setup(
    author="Henrik Finsberg",
    author_email="henriknf@simula.no",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="Library for tracking motion in cardiac mps data",
    install_requires=requirements,
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="mps_motion_tracking",
    name="mps_motion_tracking",
    packages=find_packages(include=["mps_motion_tracking", "mps_motion_tracking.*"]),
    test_suite="tests",
    url="https://github.com/ComputationalPhysiology/mps_motion_tracking",
    version="0.1.0",
    extras_require=extras_require,
    project_urls={
        "Source": "https://github.com/ComputationalPhysiology/mps_motion_tracking",
    },
    zip_safe=False,
)
