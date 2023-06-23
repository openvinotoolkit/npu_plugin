#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

set(TARGET_NAME protopipe)

find_package(gflags QUIET)
find_package(yaml-cpp QUIET)
find_package(OpenVINO QUIET COMPONENTS Runtime)
find_package(Threads QUIET)

find_package(OpenCV QUIET COMPONENTS gapi)
# FIXME: Since OpenCV might come from OpenVINO developer package
# find_package(OpenCV 4.8 QUIET COMPONENETS gapi)
# is not enough to disable tool.
if("${OpenCV_VERSION}" VERSION_LESS 4.8)
    message(STATUS "OpenCV not found or version is less than 4.8. Protopipe is disabled.")
    return()
endif()

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(DEPENDENCIES
        Threads::Threads
        gflags
        yaml-cpp
        openvino::runtime
        opencv_gapi
)

if (WIN32)
    # WA: add_tool_target expects to have all dependencies as cmake targets.
    add_library(winmm INTERFACE)
    target_link_libraries(winmm INTERFACE "winmm.lib")
    list(APPEND DEPENDENCIES winmm)
endif()

add_tool_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    ADDITIONAL_SOURCE_DIRS ${SOURCE_DIR}
    INCLUDES ${SOURCE_DIR}
    LINK_LIBRARIES ${DEPENDENCIES}
)
