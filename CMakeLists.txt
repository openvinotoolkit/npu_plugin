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

add_subdirectory(thirdparty)

# Enable MCM compiler by default
if(NOT DEFINED ENABLE_MCM_COMPILER)
    set(ENABLE_MCM_COMPILER ON)
endif()

# Disable vpualHost by default
if(NOT DEFINED ENABLE_VPUAL)
    set(ENABLE_VPUAL OFF)
endif()

add_subdirectory(src/kmb_plugin)

if(ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
