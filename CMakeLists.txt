# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 3.3)

project(InferenceEngineKMBPlugin)

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

#FIXMIE remove movidius?
add_subdirectory(thirdparty)
#FIXME remove kmb_plugin folder?
add_subdirectory(src)

#enable_testing()

#if(ENABLE_TESTS)
#    add_subdirectory(tests)
#endif()
