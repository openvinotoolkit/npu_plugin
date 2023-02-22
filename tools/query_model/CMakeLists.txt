#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

set(TARGET_NAME query_model)

find_package(gflags QUIET)

add_tool_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDENCIES
        openvino_intel_vpux_plugin
    LINK_LIBRARIES
        gflags
        kmb_utils
        openvino::runtime
    )
