# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME HDDL2UnitTests)

file(GLOB_RECURSE HDDL2_UNIT_TESTS_SOURCES "*.cpp")

add_executable(${TARGET_NAME} ${HDDL2_UNIT_TESTS_SOURCES})

target_compile_options(${TARGET_NAME}
    PRIVATE
        -Wall
        -Werror)

target_link_libraries(${TARGET_NAME}
    PRIVATE
        HDDL2Plugin_test_static
        HddlUnite
        IE::gtest
        IE::gtest_main
        )

add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${TARGET_NAME} FOR_TARGETS ${TARGET_NAME} ALL)
endif()
