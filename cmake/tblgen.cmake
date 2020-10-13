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

include(TableGen)

get_directory_property(MLIR_TABLEGEN_EXE DIRECTORY ${MLIR_SOURCE_DIR} DEFINITION MLIR_TABLEGEN_EXE)

function(vpux_add_tblgen_command)
    set(options)
    set(oneValueArgs TOOL MODE SOURCE OUTPUT)
    set(multiValueArgs EXTRA_ARGS INCLUDES)
    cmake_parse_arguments(TBLGEN "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    file(RELATIVE_PATH ofn_rel ${PROJECT_SOURCE_DIR} ${TBLGEN_OUTPUT})

    get_filename_component(dst_dir ${TBLGEN_OUTPUT} DIRECTORY)
    file(MAKE_DIRECTORY ${dst_dir})

    set(cmd_args ${TBLGEN_MODE} ${TBLGEN_EXTRA_ARGS})
    list(APPEND cmd_args -I "${CMAKE_CURRENT_SOURCE_DIR}/tblgen")
    foreach(include_dir IN LISTS LLVM_INCLUDE_DIRS MLIR_INCLUDE_DIRS TBLGEN_INCLUDES)
        list(APPEND cmd_args -I ${include_dir})
    endforeach()

    if(IS_ABSOLUTE TBLGEN_SOURCE)
        set(LLVM_TARGET_DEFINITIONS ${TBLGEN_SOURCE})
    else()
        set(LLVM_TARGET_DEFINITIONS "${CMAKE_CURRENT_SOURCE_DIR}/${TBLGEN_SOURCE}")
    endif()
    set(TABLEGEN_OUTPUT)
    tablegen(${TBLGEN_TOOL} ${ofn_rel} ${cmd_args})

    add_custom_command(
        OUTPUT ${TBLGEN_OUTPUT}
        COMMAND ${CMAKE_COMMAND} -E copy ${TABLEGEN_OUTPUT} ${TBLGEN_OUTPUT}
        DEPENDS ${TABLEGEN_OUTPUT}
        COMMENT "[${TBLGEN_TOOL} TableGen ${TBLGEN_MODE}] ${TBLGEN_SOURCE}"
    )

    set(TBGGEN_OUTPUT_FILES ${TBGGEN_OUTPUT_FILES} ${TBLGEN_OUTPUT} PARENT_SCOPE)
endfunction()

function(vpux_add_tblgen_target TARGET_NAME)
    add_custom_target(${TARGET_NAME} ALL
        DEPENDS ${TBGGEN_OUTPUT_FILES}
        SOURCES ${ARGN}
        COMMENT "[TableGen] ${TARGET_NAME}"
    )
endfunction()
