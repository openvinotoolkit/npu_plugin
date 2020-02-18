# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Honor <LANG>_VISIBILITY_PRESET for all target types (including static libraries).
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)

# Set PROJECT_VERSION* variables by project command only.
set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)

cmake_minimum_required(VERSION 3.13.0 FATAL_ERROR)

project(InferenceEngineKMBPlugin)

set(IE_MAIN_KMB_PLUGIN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/cmake")

#
# Build rules
#

find_package(InferenceEngineDeveloperPackage REQUIRED)

include(features)

if(UNIX)
    ie_add_compiler_flags(-Wno-undef)
endif()

add_subdirectory(thirdparty)

if(ENABLE_MCM_COMPILER)
    add_definitions(-DENABLE_MCM_COMPILER)
    add_subdirectory(tools)
endif()

# TODO Fronend mcm lib should be built only if mcm compiler enabled, but now we have some
#   dependencies, like mcm_config
add_subdirectory(src/frontend_mcm)

if (ENABLE_VPUAL)
    add_definitions(-DENABLE_VPUAL)
endif()

add_subdirectory(src/utils)
add_subdirectory(src/kmb_plugin)

if (ENABLE_HDDL2)
    add_subdirectory(src/hddl2_plugin)
endif()

if(ENABLE_TESTS)
    enable_testing()

    if(ENABLE_VPUAL AND ENABLE_MCM_COMPILER)
        add_subdirectory(src/vpual_model)
    endif()

    add_subdirectory(tests)
    add_subdirectory(tests_experimental)
endif()

if(ENABLE_KMB_SAMPLES)
    add_subdirectory(samples)
endif()

# install
ie_cpack(kmb)
