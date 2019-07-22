# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 3.3)

project(InferenceEngineKMBPlugin)

####################################
## All next project will use C++11
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
####################################

set(IE_MAIN_KMB_PLUGIN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

#
# Find Inference Engine Developer package
#

find_package(InferenceEngineDeveloperPackage REQUIRED)

# resolving dependencies for the current project
include(ExternalProject)

#
# Build rules
#

# Enable vpualHost by default if target is arm, disable otherwise
if(NOT DEFINED ENABLE_VPUAL)
    set(ENABLE_VPUAL ON)
    # Try to locate external vpualHost (location can be set via vpualHost_DIR).
    # If no external vpualHost provided, it will be built from thirdparty directory
    find_package(vpualHost QUIET)
endif()

add_subdirectory(thirdparty)

# Enable MCM compiler by default
if(NOT DEFINED ENABLE_MCM_COMPILER)
    set(ENABLE_MCM_COMPILER ON)
endif()

add_subdirectory(src/kmb_plugin)

if(ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(NOT DEFINED ENABLE_KMB_SAMPLES OR ENABLE_SAMPLES)
    set(ENABLE_KMB_SAMPLES ON)
endif()

if(ENABLE_KMB_SAMPLES)
    add_subdirectory(samples)
endif()
