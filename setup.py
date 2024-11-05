# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Setup script."""

import os
import re
import subprocess
import sys


_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

_WINDOWS_DEPENDENCIES = {
    "torch": "https://download.pytorch.org/whl/torch_stable.html",
}

_IS_WINDOWS = sys.platform in ["cygwin", "win32", "windows"]


def _preinstall_requirement(requirement, options=None):
    args = ["pip", "install", requirement, *(options or [])]
    return_code = subprocess.call(args)
    if return_code != 0:
        raise RuntimeError(f"{requirement} installation failed")


def _parse_requirements(requirements_file):
    requirements_file = os.path.realpath(requirements_file)
    requirements = []
    with open(requirements_file, encoding="utf-8") as f:
        for requirement in f:
            # Ignore lines with `-f` flag
            if requirement.split(" ")[0] == "-f":
                continue
            if not _IS_WINDOWS:
                requirements.append(requirement)
                continue
            package_name = re.split("[=|<>!~]", requirement)[0]
            if package_name not in _WINDOWS_DEPENDENCIES:
                requirements.append(requirement)
                continue
            # Windows-specific requirement
            url = _WINDOWS_DEPENDENCIES[package_name]
            _preinstall_requirement(requirement, options=["-f", url])
    return requirements


with open(os.path.join(_ROOT_DIR, "audiocodecs", "version.py")) as f:
    tmp = {}
    exec(f.read(), tmp)
    _VERSION = tmp["VERSION"]
    del tmp

with open(os.path.join(_ROOT_DIR, "README.md"), encoding="utf-8") as f:
    _README = f.read()

_REQUIREMENTS_SETUP = _parse_requirements(
    os.path.join(_ROOT_DIR, "requirements", "requirements-setup.txt")
)

_REQUIREMENTS_BASE = _parse_requirements(
    os.path.join(_ROOT_DIR, "requirements", "requirements-base.txt")
)

_REQUIREMENTS_ALL = _parse_requirements(
    os.path.join(_ROOT_DIR, "requirements", "requirements-all.txt")
)

_REQUIREMENTS_DEV = _parse_requirements(
    os.path.join(_ROOT_DIR, "requirements", "requirements-dev.txt")
)

# Manually preinstall setup requirements since build system specification in
# pyproject.toml is not reliable. For example, when NumPy is preinstalled,
# NumPy extensions are compiled with the latest compatible NumPy version
# rather than the one available on the system. If the two NumPy versions
# do not match, a runtime error is raised
for requirement in _REQUIREMENTS_SETUP:
    _preinstall_requirement(requirement)


from setuptools import find_packages, setup  # noqa: E402


setup(
    name="audiocodecs",
    version=_VERSION,
    description="A collection of audio codecs with a standardized API",
    long_description=_README,
    long_description_content_type="text/markdown",
    author="Luca Della Libera",
    author_email="luca.dellalib@gmail.com",
    url="https://github.com/lucadellalib/audiocodecs",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=["Audio", "Codecs", "PyTorch"],
    platforms=["OS Independent"],
    include_package_data=True,
    install_requires=_REQUIREMENTS_BASE,
    extras_require={"all": _REQUIREMENTS_ALL, "dev": _REQUIREMENTS_DEV},
    python_requires=">=3.8",
)
