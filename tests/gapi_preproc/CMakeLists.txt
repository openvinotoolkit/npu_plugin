# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

find_package(OpenCV COMPONENTS gapi)
if(NOT OpenCV_FOUND)
    message(WARNING "No suitable OpenCV version detected, " ${TARGET_NAME} " skipped")
    return()
endif()

file(GLOB SOURCES *.cpp)
file(GLOB HEADERS *.hpp common/*.hpp cpu/*.hpp)

add_subdirectory(gapi_test_computations)

set(TARGET gapi_preproc_tests)
add_executable(${TARGET} ${SOURCES} ${HEADERS})

target_link_libraries(${TARGET} PRIVATE opencv_core
                                        opencv_imgproc
                                        gapi_test_computations
                                        IE::gtest IE::gtest_main
                                        kmbPlugin_test_static)

if(${GAPI_TEST_PERF})
    target_compile_options(${TARGET}
            PRIVATE
            -Wall
            -Werror)
    target_compile_definitions(${TARGET} PRIVATE -DPERF_TEST=1)
else()
  target_compile_definitions(${TARGET} PRIVATE -DPERF_TEST=0)
endif()
