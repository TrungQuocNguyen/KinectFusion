cmake_minimum_required(VERSION 2.8)
project(kinectfusion)

set(CMAKE_CXX_FLAGS "-O2 -Wall -Wextra")

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS})

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(test_dataset "./test/test_dataset.cpp")

add_executable(test_surface_measurement "./test/test_surface_measurement.cpp")

file(GLOB KERNELS "./include/cuda/*.cu" )
cuda_add_executable(test_surface_reconstruct ${KERNELS} "./test/test_surface_reconstruct.cpp")

add_executable(kinectfusion "./kinectfusion.cpp")
