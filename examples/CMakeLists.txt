add_executable(hello_mma hello_mma.cpp)

target_link_directories(hello_mma PRIVATE ${MMA_INCLUDE_PATH})
target_link_libraries(hello_mma mma)