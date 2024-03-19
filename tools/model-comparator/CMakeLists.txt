#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

set(TARGET_NAME model-comparator)

find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)
find_package(gflags QUIET)

add_tool_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    LINK_LIBRARIES
        openvino::runtime::dev
        openvino::runtime
        opencv_core opencv_imgproc opencv_imgcodecs
        gflags
        npu_llvm_utils
        npu_model_utils
)
