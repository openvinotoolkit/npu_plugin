#
# Copyright 2022 Intel Corporation.
#
# LEGAL NOTICE: Your use of this software and any required dependent software
# (the "Software Package") is subject to the terms and conditions of
# the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
# which may also include notices, disclaimers, or license terms for
# third party or open source software included in or with the Software Package,
# and your use indicates your acceptance of all such terms. Please refer
# to the "third-party-programs.txt" or other similarly-named text file
# included with the Software Package for additional details.
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
