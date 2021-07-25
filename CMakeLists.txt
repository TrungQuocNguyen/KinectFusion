cmake_minimum_required(VERSION 3.10)
project(kinectfusion)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-O3 -Wall")

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS})

include_directories(${PROJECT_SOURCE_DIR}/include/)
file(GLOB KERNELS ${PROJECT_SOURCE_DIR}/src/cuda/*.cu)

add_executable(test_dataset "./test/test_dataset.cpp")

cuda_add_executable(test_surface_measurement "./test/test_surface_measurement.cpp" ${KERNELS})

cuda_add_executable(test_surface_reconstruction "./test/test_surface_reconstruction.cpp" ${KERNELS})

cuda_add_executable(test_surface_prediction "./test/test_surface_prediction.cpp" ${KERNELS})

cuda_add_executable(test_kinectfusion "./test/test_kinectfusion.cpp" ${KERNELS})
