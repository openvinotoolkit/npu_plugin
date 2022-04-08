#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#

vpux_setup_lit_tool(flatc)

add_custom_target(copy_vpuip_schema_tests ALL
    COMMAND
        ${CMAKE_COMMAND} -E remove_directory
            "$<TARGET_FILE_DIR:vpuxUnitTests>/lit-tests/schema"
    COMMAND
        ${CMAKE_COMMAND} -E copy_directory
            "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/graphFile-schema/src/schema"
            "$<TARGET_FILE_DIR:vpuxUnitTests>/lit-tests/schema"
    COMMENT "[LIT] Copy VPUIP schema files for tests"
)
set_target_properties(copy_vpuip_schema_tests PROPERTIES FOLDER "tests")

set(vpuip_schema_file "path('../schema/graphfile.fbs')")

add_custom_target(copy_tests_data ALL
    COMMAND
        ${CMAKE_COMMAND} -E remove_directory
            "$<TARGET_FILE_DIR:vpuxUnitTests>/lit-tests/data"
    COMMAND
        ${CMAKE_COMMAND} -E copy_directory
            "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/tests/data"
            "$<TARGET_FILE_DIR:vpuxUnitTests>/lit-tests/data"
    COMMENT "[LIT] Copy tests data"
)
set_target_properties(copy_tests_data PROPERTIES FOLDER "tests")

set(profiling_0_bin "path('../data/profiling-0.bin')")

vpux_setup_lit_tests(VPUX
    PATTERNS "*.mlir" "*.json"
    EXTRA_SOURCES "lit.local.cfg"
    VARS "vpuip_schema_file" "profiling_0_bin"
    SUBSTITUTIONS "vpuip_schema_file" "profiling_0_bin"
)
