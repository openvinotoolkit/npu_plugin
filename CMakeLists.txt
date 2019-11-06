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
# TODO: can it be reused from IE package?
if(MSVC64 OR MINGW64)
  set(X86_64 1)
elseif(MINGW OR (MSVC AND NOT CMAKE_CROSSCOMPILING))
  set(X86 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
  set(X86_64 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "i686.*|i386.*|x86.*|amd64.*|AMD64.*")
  set(X86 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm.*|ARM.*)")
  set(ARM 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64.*|AARCH64.*)")
  set(AARCH64 1)
endif()

#
# Find Inference Engine Developer package
#
if (CMAKE_CROSSCOMPILING)
    set(CMAKE_STAGING_PREFIX "${CMAKE_SOURCE_DIR}/thirdparty")
endif()
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

# Enable MCM compiler by default
if(NOT DEFINED ENABLE_MCM_COMPILER AND NOT (ARM OR AARCH64))
    set(ENABLE_MCM_COMPILER ON)
endif()

if(ENABLE_MCM_COMPILER)
    add_definitions(-DENABLE_MCM_COMPILER)
endif()
add_subdirectory(thirdparty)

add_subdirectory(tools)

add_subdirectory(src/kmb_plugin)

add_subdirectory(src/utils)

####      HDDL 2
option(ENABLE_HDDL2 "Enable HDDL2 Plugin" ON)

if (ARM OR AARCH64)
    set(ENABLE_HDDL2 OFF)
endif()

if (ENABLE_HDDL2)
    add_subdirectory(src/hddl2_plugin)
endif()

####      TESTS
if(ENABLE_TESTS)
    enable_testing()
    # TODO: do not use model for ARM for now because the model requires mcm compiler which is not build on ARM
    if(ENABLE_VPUAL AND NOT (ARM OR AARCH64) AND ENABLE_MCM_COMPILER)
        add_subdirectory(src/vpual_model)
    endif()
    add_subdirectory(tests)
endif()

if(NOT DEFINED ENABLE_KMB_SAMPLES OR ENABLE_SAMPLES)
    set(ENABLE_KMB_SAMPLES ON)
endif()

if(ENABLE_KMB_SAMPLES)
    add_subdirectory(samples)
endif()
