# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "HDDL2FunctionalTests")

file(GLOB_RECURSE HDDL2_TESTS_SOURCES "*.cpp")

add_executable(${TARGET_NAME} ${HDDL2_TESTS_SOURCES})

target_include_directories(${TARGET_NAME} PRIVATE
        cases
        ${IE_MAIN_SOURCE_DIR}/include
        )

target_compile_options(${TARGET_NAME}
        PRIVATE
        -Wall
        -Wextra
        -Werror)

target_link_libraries(${TARGET_NAME} PRIVATE
        IE::IESharedTests
        HDDL2Plugin
        )
