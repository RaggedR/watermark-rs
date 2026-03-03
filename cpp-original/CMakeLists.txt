cmake_minimum_required(VERSION 3.10)
project(BlindWatermarkDetection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})

# Blind detection executable
add_executable(detect_blind
    detect_blind.cpp
    WatermarkDetection.cpp
    Utilities.cpp
)

target_link_libraries(detect_blind
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
