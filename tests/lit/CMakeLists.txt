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
            "${VPUX_ELF_SOURCE_DIR}/src/schema"
            "$<TARGET_FILE_DIR:vpuxUnitTests>/lit-tests/schema"
    COMMENT "[LIT] Copy VPUIP schema files for tests"
)
set_target_properties(copy_vpuip_schema_tests PROPERTIES FOLDER "tests")

set(vpuip_schema_file "path('../schema/graphfile.fbs')")

add_custom_target(copy_lit_tests_script ALL
    COMMAND
        ${CMAKE_COMMAND} -E copy
            "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/scripts/run_all_lit_tests.sh"
            "$<TARGET_FILE_DIR:vpuxUnitTests>/lit-tests/run_all_lit_tests.sh"
        COMMENT "[LIT] Copy runner script"
)

set(profiling_0_30XX_bin "path('VPUX30XX/data/profiling-0-30XX.bin')")
set(profiling_0_37XX_bin "path('VPUX37XX/data/profiling-0-37XX.bin')")
set(profiling_0_37XX_PLL_10_bin "path('VPUX37XX/data/profiling-0-37XX-PLL-10.bin')")
set(profiling_0_37XX_MVN_bin "path('VPUX37XX/data/profiling-0-37XX-MVN.bin')")

vpux_setup_lit_tests(VPUX
    PATTERNS "*.mlir" "*.json" "profiling*.bin"
    EXTRA_SOURCES "lit.local.cfg"
    VARS "vpuip_schema_file" "profiling_0_30XX_bin" "profiling_0_37XX_bin" "profiling_0_37XX_PLL_10_bin" "profiling_0_37XX_MVN_bin"
    PARAMS "arch"
    PARAMS_DEFAULT_VALUES "VPUX30XX"
    SUBSTITUTIONS "vpuip_schema_file" "profiling_0_30XX_bin" "profiling_0_37XX_bin" "profiling_0_37XX_PLL_10_bin" "profiling_0_37XX_MVN_bin"
)
