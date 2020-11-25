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

function(vpux_add_flatc_target FLATC_TARGET_NAME)
    set(options)
    set(oneValueArgs SRC_DIR DST_DIR)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(FLATC "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT FLATC_SRC_DIR OR NOT EXISTS "${FLATC_SRC_DIR}")
        message(FATAL_ERROR "SRC_DIR is missing or not exists")
    endif()
    if(NOT FLATC_DST_DIR)
        message(FATAL_ERROR "DST_DIR is missing")
    endif()

    file(GLOB FLATC_SOURCES "${FLATC_SRC_DIR}/*.fbs")
    source_group(TREE ${FLATC_SRC_DIR} FILES ${FLATC_SOURCES})

    file(MAKE_DIRECTORY ${FLATC_DST_DIR})

    set(dst_files)
    foreach(src_file IN LISTS FLATC_SOURCES)
        get_filename_component(file_name_we ${src_file} NAME_WE)
        set(dst_file "${FLATC_DST_DIR}/${file_name_we}_generated.h")
        list(APPEND dst_files ${dst_file})
    endforeach()

    add_custom_command(
        OUTPUT
            ${dst_files}
        COMMAND
            ${flatc_COMMAND} -o ${FLATC_DST_DIR} --cpp ${FLATC_ARGS} ${FLATC_SOURCES}
        DEPENDS
            ${FLATC_SOURCES}
            ${flatc_COMMAND}
            ${flatc_TARGET}
        COMMENT
            "[flatc] Generating schema for ${FLATC_SRC_DIR} ..."
        VERBATIM
    )

    add_custom_target(${FLATC_TARGET_NAME}
        DEPENDS
            ${dst_files}
            ${flatc_TARGET}
        SOURCES
            ${FLATC_SOURCES}
    )
endfunction()
