#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

cmake_minimum_required(VERSION 3.5.0)
project(validateTensor VERSION 0.0.1)
set(TARGET_NAME prof_parser)

find_package(gflags QUIET)

add_tool_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    LINK_LIBRARIES
        flatbuffers
        gflags
        vpux_utils
)
