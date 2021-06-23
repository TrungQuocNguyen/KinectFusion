cmake_minimum_required(VERSION 2.8)
project(kinectfusion)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(test_dataset "./test/test_dataset.cpp")
target_link_libraries(test_dataset ${OpenCV_LIBS})

add_executable(test_surface_measurement "./test/test_surface_measurement.cpp")
target_link_libraries(test_surface_measurement ${OpenCV_LIBS})

add_executable(kinectfusion "./kinectfusion.cpp")
target_link_libraries(kinectfusion ${OpenCV_LIBS})
