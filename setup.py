#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

requirements = []

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
    entry_points={
        "console_scripts": [
            "mps_motion_tracking=mps_motion_tracking.cli:main",
        ],
    },
    install_requires=requirements,
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="mps_motion_tracking",
    name="mps_motion_tracking",
    packages=find_packages(include=["mps_motion_tracking", "mps_motion_tracking.*"]),
    test_suite="tests",
    url="https://github.com/finsberg/mps_motion_tracking",
    version="0.1.0",
    project_urls={
        "Documentation": "https://mps-motion-tracking.readthedocs.io.",
        "Source": "https://github.com/finsberg/mps_motion_tracking",
    },
    zip_safe=False,
)
