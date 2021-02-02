#
# Copyright 2020 Intel Corporation.
#
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you (End User License Agreement for the Intel(R) Software
# Development Products (Version May 2017)). Unless the License provides
# otherwise, you may not use, modify, copy, publish, distribute, disclose or
# transmit this software or the related documents without Intel's prior
# written permission.
#
# This software and the related documents are provided as is, with no
# express or implied warranties, other than those that are expressly
# stated in the License.
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
