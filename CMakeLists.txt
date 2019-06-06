# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 3.3)

project(InferenceEngineKMBPlugin)

set(IE_MAIN_KMB_PLUGIN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

#
# Find Inference Engine Developer package
#

find_package(InferenceEngineDeveloperPackage REQUIRED)

# resolving dependencies for the current project
#include(dependencies)
#FIXME: needed for graph_transformer
include(ie_parallel)
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

#
# Build rules
#

add_subdirectory(src/kmb_plugin)

enable_testing()

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
