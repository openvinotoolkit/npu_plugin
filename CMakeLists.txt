#
# Copyright 2019-2020 Intel Corporation.
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

#
# Project properties
#

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

# Honor <LANG>_VISIBILITY_PRESET for all target types (including static libraries).
cmake_policy(SET CMP0063 NEW)

# Set PROJECT_VERSION* variables by project command only.
cmake_policy(SET CMP0048 NEW)

project(InferenceEngineVPUXPlugin)

#
# Build properties
#

set(VPUX_DEVICE_NAME        "VPUX")
set(KMB_DEVICE_NAME         "KMB")
set(MCM_COMPILER_COMPONENT  "mcmCompiler")

set(IE_MAIN_VPUX_PLUGIN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(THIRDPARTY_SERVER_PATH "http://nnt-srv01.inn.intel.com/dl_score_engine/")

find_package(InferenceEngineDeveloperPackage REQUIRED)

include(cmake/features.cmake)
include(cmake/compile_options.cmake)
include(cmake/dependencies.cmake)
include(cmake/cross_compilation.cmake)
include(cmake/flatbuffers.cmake)

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

ie_cpack_add_component(${VPUX_DEVICE_NAME} REQUIRED)
ie_cpack_add_component(${KMB_DEVICE_NAME} REQUIRED)

#
# Sub-directories
#

add_subdirectory(thirdparty)

set(LLVM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/llvm-project/llvm")
set(LLVM_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/thirdparty/llvm-project/llvm")

set(MLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/llvm-project/mlir")
set(MLIR_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/thirdparty/llvm-project/llvm/tools/mlir")

set(LLVM_INCLUDE_DIRS
    "${LLVM_SOURCE_DIR}/include"
    "${LLVM_BINARY_DIR}/include"
)

set(MLIR_INCLUDE_DIRS
    "${MLIR_SOURCE_DIR}/include"
    "${MLIR_BINARY_DIR}/include"
)

list(APPEND CMAKE_MODULE_PATH
    "${LLVM_SOURCE_DIR}/cmake/modules"
)

include(cmake/tblgen.cmake)

if(ENABLE_DEVELOPER_BUILD)
    add_compile_definitions(VPUX_DEVELOPER_BUILD)
endif()

add_subdirectory(src)

if(ENABLE_TESTS)
    include(cmake/lit_tests.cmake)

    add_subdirectory(tests)
    add_subdirectory(tests_deprecated)
endif()

if(ENABLE_KMB_SAMPLES)
    add_subdirectory(samples)
endif()

add_subdirectory(tools)

#
# CPack
#

set(all_components ${VPUX_DEVICE_NAME} ${KMB_DEVICE_NAME})
if(ENABLE_MCM_COMPILER_PACKAGE)
    list(APPEND all_components ${MCM_COMPILER_COMPONENT})
endif()

ie_cpack(${all_components})
