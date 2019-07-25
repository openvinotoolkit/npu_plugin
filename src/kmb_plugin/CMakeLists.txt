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

if (DEFINED MCM_COMPILER_PATHS_FINDING)
    unset(MCM_COMPILER_INTERNAL_BUILD CACHE)
    unset(MCM_COMPILER_INCLUDE_PATH CACHE)
    unset(MCM_COMPILER_LIBRARY CACHE)
    unset(FLATBUFFERRS_INCLUDE_PATH CACHE)
    unset(MCM_SCHEMA_INCLUDE_PATH CACHE)
endif()

if (ENABLE_MCM_COMPILER)
    MESSAGE(STATUS "************ Enabling MCM compiler ************")
    add_definitions(-DENABLE_MCM_COMPILER)

    if (DEFINED MCM_COMPILER_INCLUDE_PATH)
        set(MCM_COMPILER_PATHS_FINDING TRUE)
    endif()
    find_path(MCM_COMPILER_INCLUDE_PATH "include/mcm/compiler/compilation_unit.hpp"
        PATHS ENV MCM_HOME)
    find_path(FLATBUFFERRS_INCLUDE_PATH "flatbuffers/base.h"
        PATHS ENV MCM_HOME
        PATH_SUFFIXES "schema/graphfile/flatbuffers/include")
    find_path(MCM_SCHEMA_INCLUDE_PATH "graphfile_generated.h"
        PATHS ENV MCM_HOME
        PATH_SUFFIXES "meta/schema/graphfile")
    find_library(MCM_COMPILER_LIBRARY cm
        PATHS ENV MCM_HOME
        HINTS "$ENV{MCM_HOME}-build/lib" "$ENV{MCM_HOME}/build/lib")

    message(STATUS "MCM_COMPILER_INCLUDE_PATH: ${MCM_COMPILER_INCLUDE_PATH}")
    message(STATUS "MCM_COMPILER_LIBRARY: ${MCM_COMPILER_LIBRARY}")
    message(STATUS "FLATBUFFERRS_INCLUDE_PATH: ${FLATBUFFERRS_INCLUDE_PATH}")
    message(STATUS "MCM_SCHEMA_INCLUDE_PATH: ${MCM_SCHEMA_INCLUDE_PATH}")
    message(STATUS "MCM_HOME: ${MCM_HOME}")
    message(STATUS "env MCM_HOME: $ENV{MCM_HOME}")

    if(NOT MCM_COMPILER_INCLUDE_PATH OR NOT MCM_COMPILER_LIBRARY OR
       NOT FLATBUFFERRS_INCLUDE_PATH OR NOT MCM_SCHEMA_INCLUDE_PATH)
        set(MCM_BASE_DIR ${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/thirdparty/movidius/mcmCompiler)

        ExternalProject_Add(
            mcmCompiler
            PREFIX ${MCM_BASE_DIR}
            SOURCE_DIR ${MCM_BASE_DIR}
            BINARY_DIR ${MCM_BASE_DIR}/build
            STAMP_DIR ${MCM_BASE_DIR}/stamp
            INSTALL_COMMAND ""
        )

        set(MCM_COMPILER_INTERNAL_BUILD TRUE)
        set(MCM_COMPILER_INCLUDE_PATH "${MCM_BASE_DIR}")
        set(MCM_COMPILER_LIBRARY "${MCM_BASE_DIR}/build/lib/${CMAKE_SHARED_LIBRARY_PREFIX}cm${CMAKE_SHARED_LIBRARY_SUFFIX}")
        set(FLATBUFFERRS_INCLUDE_PATH "${MCM_BASE_DIR}/schema/graphfile/flatbuffers/include/")
        set(MCM_SCHEMA_INCLUDE_PATH "${MCM_BASE_DIR}/meta/schema/graphfile")
    else()
        message(STATUS "mcmCompiler is outside the IE tree")
    endif()
else()
    MESSAGE(STATUS "************ Disabling MCM compiler ************")
endif()

if (ENABLE_VPUAL)
    add_definitions(-DENABLE_VPUAL)
    set(VPUAL_NN_LIBRARY NN)
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


ie_register_plugins(MAIN_TARGET ${TARGET_NAME}
                    POSSIBLE_PLUGINS ${TARGET_NAME})

if (MCM_COMPILER_INTERNAL_BUILD)
    add_dependencies(${TARGET_NAME} mcmCompiler)
endif()

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
        )

if (MCM_COMPILER_PATHS_FINDING)
    unset(MCM_COMPILER_INTERNAL_BUILD CACHE)
    unset(MCM_COMPILER_INCLUDE_PATH CACHE)
    unset(MCM_COMPILER_LIBRARY CACHE)
    unset(MCM_COMPILER_PATHS_FINDING CACHE)
    unset(FLATBUFFERRS_INCLUDE_PATH CACHE)
    unset(MCM_SCHEMA_INCLUDE_PATH CACHE)
endif()

if(ENABLE_TESTS)
    add_library(${TARGET_NAME}_test_static STATIC ${SOURCES})

    if (MCM_COMPILER_INTERNAL_BUILD)
        add_dependencies(${TARGET_NAME}_test_static mcmCompiler)
    endif()

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
                ${VPUAL_NN_LIBRARY})
endif()
