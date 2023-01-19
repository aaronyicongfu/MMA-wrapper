cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(MMA LANGUAGES CXX)
find_package(MPI REQUIRED)

# Path variables
set(PETSC_LIB_PATH /Users/fyc/packages/petsc/lib)
set(PETSC_INCLUDE_PATH /Users/fyc/packages/petsc/include)
set(MMA_LIB_PATH ${CMAKE_SOURCE_DIR}/lib)
set(MMA_INCLUDE_PATH ${CMAKE_SOURCE_DIR}/include)
set(MMA_SRC_PATH ${CMAKE_SOURCE_DIR}/src)

# Build MMA
add_subdirectory(src)

# Build examples
add_subdirectory(examples)
