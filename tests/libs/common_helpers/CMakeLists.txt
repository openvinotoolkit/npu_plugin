#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

set(TARGET_NAME "common_tests_helpers")

add_library(${TARGET_NAME} INTERFACE)

target_include_directories(${TARGET_NAME}
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

link_system_libraries(${TARGET_NAME}
    INTERFACE
        openvino::runtime
        openvino::format_reader
)
