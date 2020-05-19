# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

find_package(OpenCV QUIET COMPONENTS core imgproc)
if(NOT OpenCV_FOUND)
    message(WARNING "No suitable OpenCV version detected, " ${TARGET_NAME} " skipped")
    return()
endif()

file(GLOB SOURCES *.cpp)
file(GLOB HEADERS *.hpp common/*.hpp cpu/*.hpp)

add_subdirectory(gapi_test_computations)

set(TARGET gapi_preproc_tests)
add_executable(${TARGET} ${SOURCES} ${HEADERS})

target_link_libraries(${TARGET}
        PRIVATE
        gapi_test_computations
        kmbPlugin_test_static
        IE::gtest IE::gtest_main
        opencv_core opencv_imgproc)

if(GAPI_TEST_PERF)
    target_compile_definitions(${TARGET} PRIVATE PERF_TEST=1)
else()
    target_compile_definitions(${TARGET} PRIVATE PERF_TEST=0)
endif()

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${TARGET} FOR_TARGETS ${TARGET} ALL)
endif()
