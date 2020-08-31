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

# Honor <LANG>_VISIBILITY_PRESET for all target types (including static libraries).
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)

# Set PROJECT_VERSION* variables by project command only.
set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)

cmake_minimum_required(VERSION 3.13.0 FATAL_ERROR)

project(InferenceEngineKMBPlugin)

set(IE_MAIN_KMB_PLUGIN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(THIRDPARTY_SERVER_PATH "http://nnt-srv01.inn.intel.com/dl_score_engine/")

list(APPEND CMAKE_MODULE_PATH "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/cmake")

#
# Build rules
#
find_package(InferenceEngineDeveloperPackage REQUIRED)

include(features)
include(compile_options)
include(dependencies)

if(UNIX)
    ie_add_compiler_flags(-Wno-undef)
endif()

print_enabled_kmb_features()

if(ENABLE_EXPORT_SYMBOLS)
    replace_compile_options()
endif()

if(ENABLE_M2I)
    add_definitions(-DENABLE_M2I)
endif()

if (ARM OR AARCH64)
    if (NOT vpualHost_DIR)
        set(vpualHost_DIR ${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/artifacts/vpualHostInstallPackage/share/vpualHost)
    endif()
    find_package(vpualHost REQUIRED CONFIG PATHS ${vpualHost_DIR} CMAKE_FIND_ROOT_PATH_BOTH)
    set(VPUSMM_LIBRARY vpusmm)
endif()

add_subdirectory(thirdparty)

add_subdirectory(src/vpux)
# TODO: Frontend mcm lib should be built only if mcm compiler enabled, but now we have some
#   dependencies, like mcm_config
add_subdirectory(src/frontend_mcm)

if(UNIX)
    add_subdirectory(src)
endif()

add_subdirectory(src/kmb_plugin)

if (ENABLE_HDDL2)
    add_subdirectory(src/hddl2_plugin)
endif()

# It should be called after all plugins to enable registration.
add_subdirectory(src/utils)

if(ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
    add_subdirectory(tests_deprecated)
endif()

add_subdirectory(tools)

if(ENABLE_KMB_SAMPLES)
    add_subdirectory(samples)
endif()

# install
ie_cpack(kmb)
