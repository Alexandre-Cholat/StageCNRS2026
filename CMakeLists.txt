# 1. Set the minimum version of CMake required
cmake_minimum_required(VERSION 3.15)
project(StageCNRS2026)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for math (highly recommended for MFCC/FFT)
if(NOT MSVC)
    add_compile_options(-O3 -Wall)
endif()

add_subdirectory(RTNeural)
add_executable(inf_architecture inf_architecture.cpp)
target_link_libraries(inf_architecture PRIVATE RTNeural)

add_subdirectory(scripts)