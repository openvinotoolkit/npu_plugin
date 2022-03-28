#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

cmake_minimum_required(VERSION 3.5.0)
project(validateTensor VERSION 0.0.1)

find_package(gflags QUIET)

add_executable(prof_parser 
    "prof_parser.cpp"
)

target_include_directories(prof_parser
    PRIVATE
        $<TARGET_PROPERTY:vpux_utils,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(prof_parser flatbuffers gflags vpux_utils)
