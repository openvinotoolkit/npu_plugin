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

set(TARGET_NAME "kmbPlugin")

if (ENABLE_MCM_COMPILER)
    MESSAGE(STATUS "************ Enabling MCM compiler ************")
    add_definitions(-DENABLE_MCM_COMPILER)

    find_path(MCM_COMPILER_INCLUDE_PATH "include/mcm/compiler/compilation_unit.hpp"
        PATHS ENV MCM_HOME)
    find_path(FLATBUFFERRS_INCLUDE_PATH "flatbuffers/base.h"
        PATHS ENV MCM_HOME
        PATH_SUFFIXES "schema/graphfile/flatbuffers/include")

    set(MCM_BASE_DIR ${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/thirdparty/movidius/mcmCompiler)
    ExternalProject_Add(
        mcmCompiler
        PREFIX ${MCM_BASE_DIR}
        SOURCE_DIR ${MCM_BASE_DIR}
        BINARY_DIR ${MCM_BASE_DIR}/build
        STAMP_DIR ${MCM_BASE_DIR}/stamp
        INSTALL_COMMAND ""
    )

    set(MCM_COMPILER_INCLUDE_PATH "${MCM_BASE_DIR}")
    set(MCM_COMPILER_LIBRARY "${MCM_BASE_DIR}/build/lib/${CMAKE_SHARED_LIBRARY_PREFIX}cm${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(MCM_COMPILER_MODEL_LIBRARY "${MCM_BASE_DIR}/build/lib/${CMAKE_SHARED_LIBRARY_PREFIX}model${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(MCM_COMPILER_METAMODEL_LIBRARY "${MCM_BASE_DIR}/build/lib/${CMAKE_SHARED_LIBRARY_PREFIX}metamodel${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(FLATBUFFERRS_INCLUDE_PATH "${MCM_BASE_DIR}/schema/graphfile/flatbuffers/include/")
    set(MCM_SCHEMA_INCLUDE_PATH "${MCM_BASE_DIR}/meta/schema/graphfile")

    # copy mcm compiler library to target directory when the compiler is enabled
    set(BINARY_OUT_DIR "$<TARGET_FILE_DIR:kmbPlugin>")
    add_custom_target(mcm_copy_libraries ALL
            COMMAND "${CMAKE_COMMAND}" -E copy "${MCM_COMPILER_LIBRARY}" "${BINARY_OUT_DIR}"
            COMMAND "${CMAKE_COMMAND}" -E copy "${MCM_COMPILER_MODEL_LIBRARY}" "${BINARY_OUT_DIR}"
            COMMAND "${CMAKE_COMMAND}" -E copy "${MCM_COMPILER_METAMODEL_LIBRARY}" "${BINARY_OUT_DIR}"
            COMMENT "[MCM_COMPILER] Copy mcm compiler libraries to build results")
else()
    MESSAGE(STATUS "************ Disabling MCM compiler ************")
endif()

if (ENABLE_VPUAL)
    add_definitions(-DENABLE_VPUAL)
    set(VPUAL_NN_LIBRARY NN)
    set(GAPI_SIPP_LIBRARY gapi_sipp)
endif()

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)

ie_add_plugin(NAME ${TARGET_NAME}
              SOURCES ${SOURCES}
              VERSION_DEFINES_FOR kmb_plugin.cpp
            DEVICE_NAME KMB)

target_compile_options(${TARGET_NAME}
        PRIVATE
        -Wall
        -Wextra
        -Werror)

# TODO: Remove when warnings are fixed in vpualHost
set_source_files_properties(kmb_preproc_gapi_kernels_sipp.cpp PROPERTIES
                            COMPILE_FLAGS "-Wno-error")

ie_register_plugins(MAIN_TARGET ${TARGET_NAME}
                    POSSIBLE_PLUGINS ${TARGET_NAME})

add_dependencies(${TARGET_NAME} mcmCompiler)

target_include_directories(${TARGET_NAME}
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
    SYSTEM PRIVATE
        "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include"
        "${IE_MAIN_SOURCE_DIR}/include"
        "${IE_MAIN_SOURCE_DIR}/src/inference_engine"
        "${MCM_COMPILER_INCLUDE_PATH}"
        "${FLATBUFFERRS_INCLUDE_PATH}"
        "${MCM_SCHEMA_INCLUDE_PATH}"
        )

target_link_libraries(${TARGET_NAME}
    PRIVATE
        ${INTEL_ITT_LIBS}
        IE::inference_engine
        IE::vpu_common_lib
        ${MCM_COMPILER_LIBRARY}
        ${VPUAL_NN_LIBRARY}
        vpusmm
        ${GAPI_SIPP_LIBRARY}
        )

if(ENABLE_TESTS)
    add_library(${TARGET_NAME}_test_static STATIC ${SOURCES})

    add_dependencies(${TARGET_NAME}_test_static mcmCompiler)

    target_include_directories(${TARGET_NAME}_test_static
        PUBLIC
            "${CMAKE_CURRENT_SOURCE_DIR}"
        SYSTEM PRIVATE
            "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include"
            "${IE_MAIN_SOURCE_DIR}/include"
            "${IE_MAIN_SOURCE_DIR}/src/inference_engine"
            "${MCM_COMPILER_INCLUDE_PATH}"
            "${FLATBUFFERRS_INCLUDE_PATH}"
            "${MCM_SCHEMA_INCLUDE_PATH}"
            )
    target_link_libraries(${TARGET_NAME}_test_static
            PUBLIC
                vpusmm
                IE::inference_engine
            PRIVATE
                ${INTEL_ITT_LIBS}
                IE::vpu_common_lib
                ${MCM_COMPILER_LIBRARY}
                ${VPUAL_NN_LIBRARY}
                ${GAPI_SIPP_LIBRARY})
endif()
