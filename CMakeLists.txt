# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.7.2 FATAL_ERROR)

project(InferenceEngineKMBPlugin)

set(IE_MAIN_KMB_PLUGIN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/cmake")

#
# Build rules
#
find_package(InferenceEngineDeveloperPackage REQUIRED)

# resolving dependencies for the current project
include(ExternalProject)

include(features)

# Enable vpualHost by default if target is arm, disable otherwise
if(ENABLE_VPUAL)
    # Try to locate external vpualHost (location can be set via vpualHost_DIR).
    # If no external vpualHost provided, it will be built from thirdparty directory
    find_package(vpualHost QUIET)
endif()

if(UNIX)
    ie_add_compiler_flags(-Wno-undef)
endif()

add_subdirectory(thirdparty)

if(ENABLE_MCM_COMPILER)
    add_definitions(-DENABLE_MCM_COMPILER)
endif()
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

    # TODO: do not use model for ARM for now because the model requires mcm compiler which is not build on ARM
    if(ENABLE_VPUAL AND NOT (ARM OR AARCH64) AND ENABLE_MCM_COMPILER)
        add_subdirectory(src/vpual_model)
    endif()

    add_subdirectory(tests)
endif()

if(ENABLE_KMB_SAMPLES)
    add_subdirectory(samples)
endif()
