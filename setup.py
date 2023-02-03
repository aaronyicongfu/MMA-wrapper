from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
import mpi4py
from glob import glob


includes = [
    "include",
    "/Users/fyc/packages/petsc/include",  # petsc headers, TODO: fix this hardcode
    "/opt/homebrew/Cellar/open-mpi/4.1.4_2/include",  # MPI TODO
    mpi4py.get_include()
]

libs = [
    "/opt/homebrew/Cellar/open-mpi/4.1.4_2/lib",
    "/Users/fyc/packages/petsc/lib",
]

ext_modules = [
    Pybind11Extension(
        "pywrapper",
        glob("src/*.cpp"),
        cxx_std=11,
        library_dirs=libs,
        libraries=["mpi", "petsc"]
    ),
]


setup(ext_modules=ext_modules, include_dirs=includes)