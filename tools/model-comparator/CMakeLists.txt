#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

set(TARGET_NAME model-comparator)

find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)
find_package(gflags QUIET)

add_tool_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    LINK_LIBRARIES
        IE::inference_engine IE::inference_engine_plugin_api IE::ngraph
        opencv_core opencv_imgproc opencv_imgcodecs
        gflags
        vpux_utils
)
