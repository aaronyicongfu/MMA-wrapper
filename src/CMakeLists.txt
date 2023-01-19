add_library(mma MMA.cc)

target_include_directories(mma PUBLIC ${MMA_INCLUDE_PATH} ${MPI_INCLUDE_PATH} ${PETSC_INCLUDE_PATH})
target_link_directories(mma PUBLIC ${PETSC_LIB_PATH})
target_link_libraries(mma PUBLIC MPI::MPI_CXX petsc)

install(TARGETS mma DESTINATION ${MMA_LIB_PATH})
