from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
import os
from os.path import join
import mpi4py

ext_modules = [
    Pybind11Extension(
        "pywrapper",
        ["src/pywrapper.cpp"],
        cxx_std=11,
    ),
]

includes = [
    "include",  # Headers of this project
    "/Users/fyc/packages/petsc/include",  # petsc headers, TODO: fix this hardcode
    "/opt/homebrew/Cellar/open-mpi/4.1.4_2/include"  # MPI TODO
]

setup(ext_modules=ext_modules, include_dirs=includes)