# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

set(TARGET_NAME benchmark_app_legacy)

find_package(gflags QUIET)
# for cross-compilation with gflags
find_package(Threads QUIET)

add_tool_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    LINK_LIBRARIES
        openvino::format_reader
        openvino::runtime
        gflags
        Threads::Threads
)

if (CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(${TARGET_NAME} PRIVATE -Wall)
endif()

find_package(OpenCV QUIET COMPONENTS core)
if(NOT OpenCV_FOUND)
    message(WARNING "OpenCV is disabled or not found, ${TARGET_NAME} will be built without OpenCV support. Set OpenCV_DIR")
else()
    target_compile_definitions(${TARGET_NAME} PRIVATE USE_OPENCV)
    target_link_libraries(${TARGET_NAME} PRIVATE opencv_core)
endif()
