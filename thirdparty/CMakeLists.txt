#
# Copyright 2019 Intel Corporation.
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

function(add_3rdparty_subdir SUBDIR_NAME)
    if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/${SUBDIR_NAME}")
        message(FATAL_ERROR "${SUBDIR_NAME} was not found. Please run `git submodule update --init --recursive`")
    endif()

    # Remove problematic compiler flags
    foreach(flag "-Werror" "-fvisibility=hidden" "-fvisibility-inlines-hidden")
        string(REPLACE ${flag} "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
        string(REPLACE ${flag} "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    endforeach()

    set(CMAKE_C_VISIBILITY_PRESET default)
    set(CMAKE_CXX_VISIBILITY_PRESET default)

    add_subdirectory(${SUBDIR_NAME})
endfunction()

add_subdirectory(vsi_cmodel/vpusmm)

if(ENABLE_VPUAL)
    # Try to locate external vpualHost (location can be set via vpualHost_DIR).
    # If no external vpualHost provided, it will be built from thirdparty directory
    find_package(vpualHost QUIET)
    if(NOT vpualHost_FOUND)
        # Including vpualHost/host directory directly not to add vpualHost's examples to the build
        add_3rdparty_subdir(movidius/vpualHost/host)
    endif()

    # copy memory manager library to target directory when vpualHost is enabled
    get_target_property(VPUSMM_LIB_PATH vpusmm IMPORTED_LOCATION)
    set(VPUSMM_OUT_DIR "$<TARGET_FILE_DIR:kmbPlugin>")
    # FIXME: dirty hack to copy vpusmm and symlinks on it
    add_custom_target(vpusmm_copy_library ALL
        COMMAND "${CMAKE_COMMAND}" -E copy "${VPUSMM_LIB_PATH}*" "${VPUSMM_OUT_DIR}"
        COMMENT "[VPUSMM] Copy vpusmm library to build results")

    add_subdirectory(gapi_sipp/sipp)

#   if(ENABLE_TESTS)
#       add_subdirectory(tests/integration)
#   endif()

    set(VPUAL_XLINK_LIBRARY XLink CACHE INTERNAL "")
    set(VPUAL_XLINK_INCLUDES $<TARGET_PROPERTY:XLink,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "")

    set(VPUAL_DISPATCHER_LIBRARY VpualDispatcher CACHE INTERNAL "")
    set(VPUAL_DISPATCHER_INCLUDES $<TARGET_PROPERTY:VpualDispatcher,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "")

    set(VPUAL_NN_LIBRARY NN CACHE INTERNAL "")
    set(VPUAL_NN_INCLUDES $<TARGET_PROPERTY:NN,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "")

    set(GAPI_SIPP_LIBRARY gapi_sipp CACHE INTERNAL "")
    set(GAPI_SIPP_INCLUDES $<TARGET_PROPERTY:gapi_sipp,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "")
else()
    set(VPUAL_XLINK_LIBRARY "" CACHE INTERNAL "")
    set(VPUAL_XLINK_INCLUDES "" CACHE INTERNAL "")

    set(VPUAL_DISPATCHER_LIBRARY "" CACHE INTERNAL "")
    set(VPUAL_DISPATCHER_INCLUDES "" CACHE INTERNAL "")

    set(VPUAL_NN_LIBRARY "" CACHE INTERNAL "")
    set(VPUAL_NN_INCLUDES "" CACHE INTERNAL "")

    set(GAPI_SIPP_LIBRARY "" CACHE INTERNAL "")
    set(GAPI_SIPP_INCLUDES "" CACHE INTERNAL "")
endif()

if(ENABLE_MCM_COMPILER)
    set(MCM_BASE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/movidius/mcmCompiler")
    set(MCM_BASE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/movidius/mcmCompiler")

    set(MCM_COMPILER_IS_SUBPROJECT TRUE)
    set(MCM_COMPILER_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    set(MCM_COMPILER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(MCM_COMPILER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_3rdparty_subdir(movidius/mcmCompiler)

    set(COMPILATION_CONFIG_OUT_DIR "$<TARGET_FILE_DIR:kmbPlugin>")
    add_custom_target(mcm_compiler_copy_configs ALL
        COMMAND "${CMAKE_COMMAND}" -E copy_directory "${MCM_BASE_SOURCE_DIR}/config" "${COMPILATION_CONFIG_OUT_DIR}/mcm_config"
        COMMENT "[MCM Compiler] Copy config files for compilation")

    set(MCM_COMPILER_LIBRARY cm CACHE INTERNAL "")
    set(MCM_COMPILER_INCLUDES $<TARGET_PROPERTY:cm,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "")
else()
    set(MCM_COMPILER_LIBRARY "" CACHE INTERNAL "")
    set(MCM_COMPILER_INCLUDES "" CACHE INTERNAL "")
endif()
