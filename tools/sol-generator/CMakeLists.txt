# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0

set(TARGET_NAME sol-generator)

find_package(gflags QUIET)

if(NOT gflags_FOUND)
    message(WARNING "${TARGET_NAME} tool is disabled due to missing dependencies")
    return()
endif()

add_executable(${TARGET_NAME} main.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE
    openvino::runtime
    gflags
)
