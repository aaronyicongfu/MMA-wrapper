add_library(MMA MMA.cc)
target_include_directories(MMA PUBLIC ${MMA_INCLUDE_PATH} ${MPI_INCLUDE_PATH} ${PETSC_INCLUDE_PATH})
target_link_directories(MMA PUBLIC ${PETSC_LIB_PATH})
target_link_libraries(MMA PUBLIC MPI::MPI_CXX petsc)

add_library(optimizer optimizer.cpp)
target_include_directories(optimizer PUBLIC ${MMA_INCLUDE_PATH} ${MPI_INCLUDE_PATH} ${PETSC_INCLUDE_PATH})
target_link_directories(optimizer PUBLIC ${PETSC_LIB_PATH})
target_link_libraries(optimizer PUBLIC MPI::MPI_CXX petsc)

install(TARGETS MMA DESTINATION ${MMA_LIB_PATH})
install(TARGETS optimizer DESTINATION ${MMA_LIB_PATH})
