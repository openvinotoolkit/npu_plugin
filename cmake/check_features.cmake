# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

#64 bits platform
if ("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
    message(STATUS "Detected 64 bit architecture")
    SET(ARCH_64 ON)
    SET(ARCH_32 OFF)
else()
    message(STATUS "Detected 32 bit architecture")
    SET(ARCH_64 OFF)
    SET(ARCH_32 ON)
endif()

if (NOT ARCH_64)
    if (UNIX OR APPLE)
        SET(ENABLE_CLDNN OFF)
    endif()
    SET(ENABLE_MKL_DNN OFF)
    SET(ENABLE_ICV_TESTS OFF)
    SET(ENABLE_DLIA OFF)
    SET(ENABLE_HDDL OFF)
endif()

#apple specific
if (APPLE)
    set(ENABLE_GNA OFF)
    set(ENABLE_CLDNN OFF)
    SET(ENABLE_DLIA OFF)
    SET(ENABLE_MYRIAD OFF)
    SET(ENABLE_HDDL OFF)
    SET(ENABLE_KMB OFF)
endif()


#minGW specific - under wine no support for downloading file and applying them using git
if (WIN32)
    if (MINGW)
        SET(ENABLE_CLDNN OFF) # dont have mingw dll for linking
        set(ENABLE_SAMPLES OFF)
    endif()
    if (ENABLE_KMB)
        # kmbPlugin doesn't build under Windows yet
        set(ENABLE_KMB OFF)
    endif()
endif()

if (NOT ENABLE_MKL_DNN)
    set(ENABLE_MKL OFF)
endif()

if (NOT ENABLE_VPU)
    set(ENABLE_MYRIAD OFF)
    SET(ENABLE_HDDL OFF)
    SET(ENABLE_KMB OFF)
endif()

if (NOT ENABLE_MYRIAD AND NOT ENABLE_HDDL AND NOT ENABLE_KMB)
    set(ENABLE_VPU OFF)
endif()

#next section set defines to be accesible in c++/c code for certain feature
if (ENABLE_PROFILING_RAW)
    add_definitions(-DENABLE_PROFILING_RAW=1)
endif()

if (ENABLE_CLDNN)
    add_definitions(-DENABLE_CLDNN=1)
endif()

if (ENABLE_DLIA)
    add_definitions(-DENABLE_DLIA=1)
endif()

if (ENABLE_MYRIAD)
    add_definitions(-DENABLE_MYRIAD=1)
endif()

if (ENABLE_MYRIAD_NO_BOOT AND ENABLE_MYRIAD )
    add_definitions(-DENABLE_MYRIAD_NO_BOOT=1)
endif()

if (ENABLE_HDDL)
    add_definitions(-DENABLE_HDDL=1)
endif()

if (ENABLE_KMB)
    add_definitions(-DENABLE_KMB=1)
endif()

if (ENABLE_MKL_DNN)
    add_definitions(-DENABLE_MKL_DNN=1)
endif()

#tests supermacro
#TODO: create ie_test_option to properly mark option to be test dependend
if (NOT ENABLE_TESTS)
    SET(ENABLE_BEH_TESTS OFF)
    SET(ENABLE_FUNCTIONAL_TESTS OFF)
    SET(ENABLE_ICV_TESTS OFF)
endif()

if (ENABLE_GNA)
    add_definitions(-DENABLE_GNA)
endif()

if (ENABLE_SAMPLES)
    set (ENABLE_SAMPLES_CORE ON)
endif()

#models dependend tests

if (DEVELOPMENT_PLUGIN_MODE)
    message (STATUS "Enabled development plugin mode")

    set (ENABLE_MKL_DNN OFF)
    set (ENABLE_TESTS OFF)

    message (STATUS "Initialising submodules")
    execute_process (COMMAND git submodule update --init ${IE_MAIN_SOURCE_DIR}/thirdparty/pugixml
                     RESULT_VARIABLE git_res)

    if (NOT ${git_res})
        message (STATUS "Initialising submodules - done")
    endif()
endif()

if (NOT ENABLE_TESTS)
    set(ENABLE_MODELS OFF)
    set(ENABLE_GNA_MODELS OFF)
endif ()

if (VERBOSE_BUILD)
    set(CMAKE_VERBOSE_MAKEFILE  ON)
endif()


if(ENABLE_DUMP)
    add_definitions(-DDEBUG_DUMP)
endif()


print_enabled_features()
