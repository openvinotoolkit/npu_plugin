#
# Copyright (C) 2022-2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

set(TARGET_NAME prof_parser)

find_package(gflags QUIET)

add_tool_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    LINK_LIBRARIES
        flatbuffers
        gflags
        npu_llvm_utils
        npu_profiling_utils
)

