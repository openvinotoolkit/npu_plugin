# Copyright (C) 2022 Intel Corporation
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

if(${CMAKE_VERSION} VERSION_LESS "3.19.0") 
    message(Warning "Source code package will not be generated."
                    "Miminum CMake version required - 3.19.0")
    return()
endif()


function(read_config_file CONFIG_FILE EXCLUDE_PATTERNS)
    # read json file
    file(READ ${CONFIG_FILE} JSON_CONFIG_STR)

    # utility function to transform a json list to a cmake list
    # and replace ^ in the resulting list's patterns
    function(config_patterns_to_cmake_list CFG_PATTERNS CFG_PATTERNS_LENGTH OUTPUT_LIST)
        set(RESULT_LIST "")
        foreach(IDX RANGE ${CFG_PATTERNS_LENGTH})
            if(NOT ${IDX} STREQUAL ${CFG_PATTERNS_LENGTH})
                string(JSON PATTERN GET ${CFG_PATTERNS} ${IDX})  
                string(REPLACE "^" "${CMAKE_SOURCE_DIR}" PATTERN ${PATTERN})
                list(APPEND RESULT_LIST ${PATTERN})
            endif()
        endforeach()

        set(${OUTPUT_LIST} ${RESULT_LIST} PARENT_SCOPE)
    endfunction()

    # read the `exclude_patterns` json variable
    string(JSON CFG_EXCLUDE_PATTERNS GET ${JSON_CONFIG_STR} "ignore_files")
    string(JSON CFG_EXCLUDE_PATTERNS_LEN LENGTH ${JSON_CONFIG_STR} "ignore_files")
    config_patterns_to_cmake_list(
        "${CFG_EXCLUDE_PATTERNS}"
        "${CFG_EXCLUDE_PATTERNS_LEN}"
        CFG_EXCLUDE_PATTERNS_LIST
    )
    
    list(APPEND CFG_EXCLUDE_PATTERNS_LIST "${CMAKE_BINARY_DIR}")

    # return variables
    set(${EXCLUDE_PATTERNS} ${CFG_EXCLUDE_PATTERNS_LIST} PARENT_SCOPE)
endfunction()

#
# setup source package configuration
#

read_config_file("${CMAKE_CURRENT_SOURCE_DIR}/SourcePackageConfig.json" EXCLUDE_PATTERNS)

set(CPACK_SOURCE_OUTPUT_CONFIG_FILE "CPackSourceConfigVPUX.cmake")
set(CPACK_SOURCE_IGNORE_FILES ${EXCLUDE_PATTERNS})
