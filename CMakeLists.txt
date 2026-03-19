# 1. Set the minimum version of CMake required
cmake_minimum_required(VERSION 3.15)

# 2. Name your project (matches your folder)
project(StageCNRS2026)

# 3. RTNeural requires modern C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 4. Tell CMake where the RTNeural folder is
add_subdirectory(RTNeural)

# 5. Build your executable from your new .cpp file
# Format: add_executable(NameOfExeYouWantToMake NameOfYourCodeFile.cpp)
add_executable(inf_architecture inf_architecture.cpp)

# 6. "Link" the RTNeural library to your executable
target_link_libraries(inf_architecture PRIVATE RTNeural)