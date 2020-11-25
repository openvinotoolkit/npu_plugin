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

set(TARGET_NAME "vpux_utils_vpuip")
set(SCHEMA_TARGET_NAME "${TARGET_NAME}_schema")

#
# Schema target
#

vpux_add_flatc_target(${SCHEMA_TARGET_NAME}
    SRC_DIR "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/graphFile-schema/src/schema"
    DST_DIR "${CMAKE_CURRENT_BINARY_DIR}/include/vpux/utils/VPUIP/schema/generated"
    ARGS "--gen-name-strings"
)
set_target_properties(${SCHEMA_TARGET_NAME} PROPERTIES FOLDER "src")

#
# Main target
#

file(GLOB_RECURSE SOURCES "*.cpp" "*.hpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} INTERFACE)
# set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "src")

add_dependencies(${TARGET_NAME} ${SCHEMA_TARGET_NAME})

ie_faster_build(${TARGET_NAME}
    # UNITY
    PCH INTERFACE "src/precomp.hpp"
)

# enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
# vpux_enable_clang_format(${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_BINARY_DIR}/include"
)

link_system_libraries(${TARGET_NAME}
    INTERFACE
        flatbuffers
)
target_link_libraries(${TARGET_NAME}
    INTERFACE
        vpux_utils_core
        vpux_utils_ie
)
