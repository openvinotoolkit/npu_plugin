#
# Copyright (C) 2022-2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#
# Project properties
#

set(CMAKE_POLICY_DEFAULT_CMP0115 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0116 OLD)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

# Honor <LANG>_VISIBILITY_PRESET for all target types (including static libraries).
cmake_policy(SET CMP0063 NEW)

# Set PROJECT_VERSION* variables by project command only.
cmake_policy(SET CMP0048 NEW)

project(InferenceEngineVPUXPlugin)

#
# Build properties
#

set(NPU_DEVICE_NAME "NPU")
string(TOLOWER "${NPU_DEVICE_NAME}" VPUX_PLUGIN_COMPONENT)
set(VPUX_TESTS_COMPONENT "${VPUX_PLUGIN_COMPONENT}_tests")

set(IE_MAIN_VPUX_PLUGIN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_VPUX_PLUGIN_BINARY_DIR})
    set(CMAKE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_VPUX_PLUGIN_BINARY_DIR})
endif()

if(DEFINED ENV{THIRDPARTY_SERVER_PATH})
    set(THIRDPARTY_SERVER_PATH "$ENV{THIRDPARTY_SERVER_PATH}")
elseif(DEFINED THIRDPARTY_SERVER_PATH)
    set(THIRDPARTY_SERVER_PATH "${THIRDPARTY_SERVER_PATH}")
endif()

include(FetchContent)

find_package(InferenceEngineDeveloperPackage REQUIRED)
get_directory_property(defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
set(IE_BUILD_POSTFIX_VAR)
foreach(def ${defs})
    if(def MATCHES "IE_BUILD_POSTFIX=")
        string(REPLACE IE_BUILD_POSTFIX= "" IE_BUILD_POSTFIX_VAR ${def})
        remove_definitions(-D${def})
    endif()
endforeach()

include(cmake/features.cmake)
include(cmake/compile_options.cmake)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies.cmake")
    include(cmake/dependencies.cmake)
endif()

include(cmake/cross_compilation.cmake)
include(cmake/flatbuffers.cmake)
include(cmake/bundle_static_library.cmake)
include(cmake/embed_bin_file.cmake)
include(cmake/add_tool_target.cmake)
include(cmake/coverage.cmake)

if(ENABLE_TESTS)
    enable_testing()
endif()

print_enabled_kmb_features()

#
# Build configuration
#

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(UNIX)
    ie_add_compiler_flags(-Wno-undef)
endif()

if(ENABLE_EXPORT_SYMBOLS)
    replace_compile_visibility_options()
endif()

#
# Sub-directories
#

add_subdirectory(thirdparty EXCLUDE_FROM_ALL)

set(VPUNN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/vpucostmodel")
set(VPUNN_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/thirdparty/vpucostmodel")

set(VPUNN_INCLUDE_DIRS
        "${VPUNN_SOURCE_DIR}/include"
        "${VPUNN_BINARY_DIR}/include"
)

if(ENABLE_DEVELOPER_BUILD)
    add_compile_definitions(VPUX_DEVELOPER_BUILD)
endif()

if(ENABLE_MLIR_COMPILER)
    add_subdirectory(sw_runtime_kernels/kernels)
endif()

add_subdirectory(src)

if(ENABLE_TESTS)
    include(cmake/lit_tests.cmake)

    add_subdirectory(tests)
endif()

add_subdirectory(tools)

#
# targets install
#

if(ENABLE_SOURCE_PACKAGE)
    include(cmake/source_package.cmake)
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION "runtime/include/ie"
        COMPONENT ${VPUX_PLUGIN_COMPONENT})

if(NOT BUILD_SHARED_LIBS)
    if(TARGET ze_loader)
        install(TARGETS ze_loader EXPORT OpenVINOTargets
                RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT}
                ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT}
                LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT})
    endif()

    if(TARGET flatbuffers)
        install(TARGETS flatbuffers EXPORT OpenVINOTargets
                RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT}
                ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT}
                LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT})
    endif()

    if(TARGET vpux_elf)
        install(TARGETS vpux_elf EXPORT OpenVINOTargets
                RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT}
                ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT}
                LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT})
    endif()
endif()

#
# CPack
#

ie_cpack(${VPUX_PLUGIN_COMPONENT})
