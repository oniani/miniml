#!/usr/bin/env python3
import setuptools


def read_long_description(path: str) -> str:
    """Utility function to read the README file."""

    with open(path) as file:
        data: str = file.read()

    return data


def read_requirements(path: str) -> list[str]:
    """Requirements file."""

    with open(path) as file:
        data: list[str] = file.readlines()

    return data


if __name__ == "__main__":
    setuptools.setup(
        name="miniml",
        version="1.0",
        author="David Oniani",
        author_email="onianidavid@gmail.com",
        description="A minimal ML library with OpenCL GPU support",
        license="MIT",
        url="https://github.com/oniani/miniml",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
        ],
        install_requires=read_requirements("requirements.txt"),
        long_description=read_long_description("README.md"),
        packages=["miniml"],
        include_package_data=True,
        zip_safe=True,
    )
