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

set(plugin_target "kmbPlugin")

include(Fetch_kmb_dependencies)

set(TARGET_NAME "kmbPlugin")

set(VPUAL_DEFAULT_BASE "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/thirdparty/movidius/vpualHost")

    if (DEFINED MCM_COMPILER_PATHS_FINDING)
        unset(MCM_COMPILER_INTERNAL_BUILD CACHE)
        unset(MCM_COMPILER_INCLUDE_PATH CACHE)
        unset(MCM_COMPILER_LIBRARY CACHE)
        unset(FLATBUFFERRS_INCLUDE_PATH CACHE)
        unset(MCM_SCHEMA_INCLUDE_PATH CACHE)
    endif()

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
        message(STATUS "Build mcmCompiler inside the IE tree")

        file(STRINGS mcmcompiler_hash.txt MCM_COMPILER_HASH)

        fetch_mcmCompiler(${MCM_COMPILER_HASH})

        set(MCM_COMPILER_INTERNAL_BUILD TRUE)
        set(MCM_COMPILER_INCLUDE_PATH "${MCM_BASE_DIR}/src/mcmCompiler")
        set(MCM_COMPILER_LIBRARY "${MCM_BASE_DIR}/src/mcmCompiler-build/lib/${CMAKE_SHARED_LIBRARY_PREFIX}cm${CMAKE_SHARED_LIBRARY_SUFFIX}")
        set(FLATBUFFERRS_INCLUDE_PATH "${MCM_BASE_DIR}/src/mcmCompiler/schema/graphfile/flatbuffers/include/")
        set(MCM_SCHEMA_INCLUDE_PATH "${MCM_BASE_DIR}/src/mcmCompiler/meta/schema/graphfile")
    else()
        message(STATUS "mcmCompiler is outside the IE tree")
    endif()

    if (NOT DEFINED ENV{VPUAL_HOME})
        if(NOT EXISTS "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/thirdparty/movidius/vpualHost")
            file(STRINGS vpual_hash.txt VPUAL_HASH)
            fetch_vpual(${VPUAL_HASH})
        endif()
        set(vpual_home "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/thirdparty/movidius/vpualHost")
    else()
        set(vpual_home $ENV{VPUAL_HOME})
    endif()

    add_definitions(-D__PC__)
    add_definitions(-DUSE_USB_VSC)

#       It works only if XLink for myriad/HDDL plugins excluded
#        include(${vpual_home}/host/source/FLIC/FLIC.cmake)
#        include(${vpual_home}/host/source/SIPP/SIPP.cmake)
#        set(VPUAL_LIBRARY VpualDispatcher RemoteFlic XLink sipp_custom)
#       Attempt to have two XLinks simultaneously
    include(vpualPrivateCmake/FLIC.cmake)
    include(vpualPrivateCmake/SIPP.cmake)
    set(VPUAL_LIBRARY RemoteFlic VpualDispatcher XLink2 sipp_custom)

    include_directories(
        ${REMOTE_FLIC_INCLUDE_DIRECTORIES}
        ${SIPP_INCLUDE_DIRECTORIES}
    )


file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)

ie_add_plugin(NAME ${TARGET_NAME}
              SOURCES ${SOURCES}
              VERSION_DEFINES_FOR kmb_plugin.cpp)

ie_register_plugins(MAIN_TARGET ${TARGET_NAME}
                    POSSIBLE_PLUGINS ${TARGET_NAME})

if (MCM_COMPILER_INTERNAL_BUILD)
    add_dependencies(${TARGET_NAME} mcmCompiler)
endif()

target_include_directories(${TARGET_NAME}
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
    SYSTEM PRIVATE
        "${IE_MAIN_SOURCE_DIR}/include"
        "${IE_MAIN_SOURCE_DIR}/src/inference_engine"
        "${MCM_COMPILER_INCLUDE_PATH}"
        "${FLATBUFFERRS_INCLUDE_PATH}"
        "${MCM_SCHEMA_INCLUDE_PATH}"
        )

# TODO: enable some day and fix all warnings
 if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    SET_TARGET_PROPERTIES( ${VPUAL_LIBRARY} PROPERTIES COMPILE_OPTIONS "-Wno-error")
 endif()
 #FIXME:
 SET_TARGET_PROPERTIES( ${TARGET_NAME} PROPERTIES COMPILE_OPTIONS "-Wno-error")

SET_TARGET_PROPERTIES( ${VPUAL_LIBRARY} PROPERTIES CXX_VISIBILITY_PRESET default )
SET_TARGET_PROPERTIES( ${VPUAL_LIBRARY} PROPERTIES C_VISIBILITY_PRESET default )

target_link_libraries(${TARGET_NAME}
    PRIVATE
  ${INTEL_ITT_LIBS} IE::inference_engine vpu_graph_transformer ${MCM_COMPILER_LIBRARY} ${VPUAL_LIBRARY})

if (MCM_COMPILER_PATHS_FINDING)
    unset(MCM_COMPILER_INTERNAL_BUILD CACHE)
    unset(MCM_COMPILER_INCLUDE_PATH CACHE)
    unset(MCM_COMPILER_LIBRARY CACHE)
    unset(MCM_COMPILER_PATHS_FINDING CACHE)
    unset(FLATBUFFERRS_INCLUDE_PATH CACHE)
    unset(MCM_SCHEMA_INCLUDE_PATH CACHE)
endif()
