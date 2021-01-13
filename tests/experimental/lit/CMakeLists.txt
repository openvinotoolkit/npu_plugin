#
# Copyright 2020 Intel Corporation.
#
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you (End User License Agreement for the Intel(R) Software
# Development Products (Version May 2017)). Unless the License provides
# otherwise, you may not use, modify, copy, publish, distribute, disclose or
# transmit this software or the related documents without Intel's prior
# written permission.
#
# This software and the related documents are provided as is, with no
# express or implied warranties, other than those that are expressly
# stated in the License.
#

add_custom_target(copy_vpuip_schema_tests ALL
    COMMAND
        ${CMAKE_COMMAND} -E remove_directory
            "$<TARGET_FILE_DIR:vpuxUnitTestsExperimental>/lit-tests/schema"
    COMMAND
        ${CMAKE_COMMAND} -E copy_directory
            "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/graphFile-schema/src/schema"
            "$<TARGET_FILE_DIR:vpuxUnitTestsExperimental>/lit-tests/schema"
    COMMENT "[LIT] Copy VPUIP schema files for tests"
)
set_target_properties(copy_vpuip_schema_tests PROPERTIES FOLDER "tests")

set(vpuip_schema_file "path('../schema/graphfile.fbs')")
vpux_setup_lit_tests(VPUX
    PATTERNS "*.mlir" "*.xml"
    SUBSTITUTIONS vpuip_schema_file
)
