#!/usr/bin/env python3
import os
from setuptools import setup


def read_long_description(filename: str) -> str:
    """Utility function to read the README file."""

    with open(os.path.join(os.path.dirname(__file__), filename)) as file:
        data: str = file.read()

    return data


def read_requirements(filename: str) -> list[str]:
    """Requirements file."""

    with open(os.path.join(os.path.dirname(__file__), filename)) as file:
        data: list[str] = file.readlines()

    return data


if __name__ == "__main__":
    setup(
        name="miniml",
        version="0.1.0",
        author="David Oniani",
        author_email="onianidavid@gmail.com",
        description="A minimal machine learning library with OpenCL GPU support",
        license="GPLv3",
        url="https://github.com/oniani/miniml",
        packages=["miniml", "tests"],
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        ],
        long_description=read_long_description("README.md"),
        install_requires=read_requirements("requirements.txt"),
    )
