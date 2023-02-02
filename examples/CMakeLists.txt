add_executable(example example.cpp)

target_link_directories(example PRIVATE ${MMA_INCLUDE_PATH})
target_link_libraries(example MMA)
target_link_libraries(example optimizer)