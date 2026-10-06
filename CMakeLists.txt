cmake_minimum_required(VERSION 3.16)
project(rgbd_recognition CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(rgbdcommon src/common.cpp)
target_link_libraries(rgbdcommon ${OpenCV_LIBS})

add_executable(train_eval src/train_eval.cpp)
target_link_libraries(train_eval rgbdcommon ${OpenCV_LIBS})

add_executable(scene_eval src/scene_eval.cpp)
target_link_libraries(scene_eval rgbdcommon ${OpenCV_LIBS})

add_executable(tracking src/tracking.cpp)
target_link_libraries(tracking rgbdcommon ${OpenCV_LIBS})
add_executable(visualize src/visualize.cpp)
target_link_libraries(visualize rgbdcommon ${OpenCV_LIBS})
