#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

set(TARGET_NAME test_classification)

find_package(gflags QUIET)

if(NOT gflags_FOUND)
    message(WARNING "${TARGET_NAME} tool is disabled due to missing dependencies")
    return()
endif()

add_executable(${TARGET_NAME} main.cpp classification_results_old.h)

target_link_libraries(${TARGET_NAME} PRIVATE
    IE::inference_engine IE::ngraph IE::format_reader
    gflags
    kmb_utils
    IE::ie_samples_utils)
