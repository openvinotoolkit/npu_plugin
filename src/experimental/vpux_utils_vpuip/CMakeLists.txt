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

set(schema_src_dir "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/graphFile-schema/src/schema")

file(GLOB FBS_SOURCES "${schema_src_dir}/*.fbs")
source_group(TREE ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR} FILES ${FBS_SOURCES})

set(schema_dst_dir "${CMAKE_CURRENT_BINARY_DIR}/include/vpux/utils/VPUIP/schema/generated")
file(MAKE_DIRECTORY ${schema_dst_dir})

set(schema_dst_files)
foreach(src_file IN LISTS FBS_SOURCES)
    get_filename_component(file_name_we ${src_file} NAME_WE)
    set(dst_file "${schema_dst_dir}/${file_name_we}_generated.h")
    list(APPEND schema_dst_files ${dst_file})
endforeach()

# TODO: Add `--scoped-enums` once schema is fixed
add_custom_command(
    OUTPUT
        ${schema_dst_files}
    COMMAND
        ${flatc_COMMAND} -o ${schema_dst_dir} --cpp --gen-name-strings ${FBS_SOURCES}
    DEPENDS
        ${FBS_SOURCES}
        ${flatc_COMMAND}
    COMMENT
        "[flatc] Generating VPUIP schema ..."
    VERBATIM
)

add_custom_target(${SCHEMA_TARGET_NAME}
    DEPENDS
        ${schema_dst_files}
    SOURCES
        ${FBS_SOURCES}
)
set_target_properties(${SCHEMA_TARGET_NAME} PROPERTIES FOLDER "src")

if(TARGET BUILD_flatc)
    add_dependencies(${SCHEMA_TARGET_NAME} BUILD_flatc)
endif()

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
