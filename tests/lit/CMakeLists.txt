#
# Copyright Intel Corporation.
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

vpux_setup_lit_tests(VPUX
    PATTERNS "*.mlir"
    EXTRA_SOURCES "lit.local.cfg"
    VARS "vpuip_schema_file"
    SUBSTITUTIONS "vpuip_schema_file"
)
