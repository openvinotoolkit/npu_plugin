#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

set(TARGET_NAME test_classification)

find_package(gflags QUIET)

add_tool_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}
    LINK_LIBRARIES
        IE::inference_engine IE::ngraph IE::format_reader
        gflags
        kmb_utils
        IE::ie_samples_utils
)
