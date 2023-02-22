# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
function(vpux_embed_bin_file)
    set(options APPEND)
    set(oneValueArgs SOURCE_FILE HEADER_FILE VARIABLE_NAME)
    set(multiValueArgs)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_SOURCE_FILE)
        message(FATAL_ERROR "Missing SOURCE_FILE argument in vpux_embed_bin_file")
    endif()
    if(NOT ARG_HEADER_FILE)
        message(FATAL_ERROR "Missing HEADER_FILE argument in vpux_embed_bin_file")
    endif()
    if(NOT ARG_VARIABLE_NAME)
        message(FATAL_ERROR "Missing VARIABLE_NAME argument in vpux_embed_bin_file")
    endif()

    if(NOT EXISTS ${ARG_SOURCE_FILE})
        message(FATAL_ERROR "File '${ARG_SOURCE_FILE}' does not exist")
    endif()

    file(READ ${ARG_SOURCE_FILE} hex_string HEX)
    string(LENGTH "${hex_string}" hex_string_length)

    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "static_cast<char>(0x\\1), " hex_array "${hex_string}")
    math(EXPR hex_array_size "${hex_string_length} / 2")
    
    if (hex_array_size LESS "1000") 
        message(FATAL_ERROR "File '${ARG_SOURCE_FILE}' too small, check that git-lfs pull step has been done.")
    endif()

    set(content "
const char ${ARG_VARIABLE_NAME}[] = { ${hex_array} };
const size_t ${ARG_VARIABLE_NAME}_SIZE = ${hex_array_size};
")

    if(ARG_APPEND)
        file(APPEND ${ARG_HEADER_FILE} "${content}")
    else()
        file(WRITE ${ARG_HEADER_FILE} "${content}")
    endif()
endfunction()
