#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#

set(TARGET_NAME single-image-test)

find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)
find_package(gflags QUIET)

if(NOT OpenCV_FOUND OR NOT gflags_FOUND)
    message(WARNING "${TARGET_NAME} tool is disabled due to missing dependencies")
    return()
endif()

add_executable(${TARGET_NAME} main.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE
    IE::inference_engine IE::ngraph
    IE::inference_engine_plugin_api
    opencv_core opencv_imgproc opencv_imgcodecs
    gflags
    kmb_utils
)
