# Copyright (C) 2019-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(COMMAND get_linux_name)
    get_linux_name(LINUX_OS_NAME)
endif()

if(NOT ENABLE_TESTS)
    set(ENABLE_TESTS OFF)
endif()
ie_dependent_option(ENABLE_TESTS "Unit, behavior and functional tests" ${ENABLE_TESTS} "TRUE" OFF)
if(ENABLE_TESTS)
    add_definitions(-DENABLE_TESTS)
endif()

if(NOT ENABLE_LTO)
    set(ENABLE_LTO OFF)
endif()
ie_dependent_option(ENABLE_LTO "Enable Link Time Optimization" ${ENABLE_LTO} "LINUX OR WIN32;NOT CMAKE_CROSSCOMPILING" OFF)

if(NOT ENABLE_FASTER_BUILD)
    set(ENABLE_FASTER_BUILD OFF)
endif()
ie_dependent_option(ENABLE_FASTER_BUILD "Enable build features (PCH, UNITY) to speed up build time" ${ENABLE_FASTER_BUILD} "CMAKE_VERSION VERSION_GREATER_EQUAL 3.16" OFF)

if(NOT ENABLE_CPPLINT)
    set(ENABLE_CPPLINT OFF)
endif()
ie_dependent_option(ENABLE_CPPLINT "Enable cpplint checks during the build" ${ENABLE_CPPLINT} "UNIX;NOT ANDROID" OFF)

if(NOT ENABLE_CLANG_FORMAT)
    set(ENABLE_CLANG_FORMAT OFF)
endif()
ie_option(ENABLE_CLANG_FORMAT "Enable clang-format checks during the build" ${ENABLE_CLANG_FORMAT})

ie_dependent_option(ENABLE_KMB_SAMPLES "Enable KMB samples" ON "AARCH64" OFF)

set(HAVE_HDDL_UNITE_PACKAGE FALSE)
if(X86_64)
    if(WIN32)
        set(HAVE_HDDL_UNITE_PACKAGE TRUE)
    elseif(LINUX)
        if(LINUX_OS_NAME STREQUAL "Ubuntu 18.04")
            set(HAVE_HDDL_UNITE_PACKAGE TRUE)
        endif()
    endif()
endif()
ie_dependent_option(ENABLE_HDDL2 "Enable HDDL2 Plugin" ON "HAVE_HDDL_UNITE_PACKAGE" OFF)
ie_dependent_option(ENABLE_HDDL2_TESTS "Enable Unit and Functional tests for HDDL2 Plugin" ON "ENABLE_HDDL2;ENABLE_TESTS" OFF)
ie_option(ENABLE_CUSTOM_HDDLUNITE "Use custom build hddlunite" OFF)

ie_dependent_option(ENABLE_MODELS "download all models required for functional testing" ON "ENABLE_FUNCTIONAL_TESTS" OFF)
ie_dependent_option(ENABLE_VALIDATION_SET "download validation_set required for functional testing" ON "ENABLE_FUNCTIONAL_TESTS" OFF)

ie_option(ENABLE_EXPORT_SYMBOLS "Enable compiler -fvisibility=default and linker -export-dynamic options" OFF)

ie_option(ENABLE_MCM_COMPILER_PACKAGE "Enable build of separate mcmCompiler package" OFF)

ie_dependent_option(ENABLE_ZEROAPI_BACKEND "Enable zero-api as a plugin backend" ON "WIN32" OFF)

ie_option(ENABLE_DEVELOPER_BUILD "Enable developer build with extra validation/logging functionality" OFF)

if(NOT DEFINED MV_TOOLS_PATH AND DEFINED ENV{MV_TOOLS_DIR} AND DEFINED ENV{MV_TOOLS_VERSION})
    set(MV_TOOLS_PATH $ENV{MV_TOOLS_DIR}/$ENV{MV_TOOLS_VERSION})
endif()
ie_dependent_option(ENABLE_EMULATOR "Enable emulator as a plugin backend" ON "ENABLE_TESTS;MV_TOOLS_PATH" OFF)

function (print_enabled_kmb_features)
    message(STATUS "KMB Plugin enabled features: ")
    message(STATUS "")
    foreach(var IN LISTS IE_OPTIONS)
        message(STATUS "    ${var} = ${${var}}")
    endforeach()
    message(STATUS "")
endfunction()
