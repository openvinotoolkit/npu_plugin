#
# Copyright 2019 Intel Corporation.
#
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you (End User License Agreement for the Intel(R) Software
# Development Products (Version May 2017)). Unless the License provides
# otherwise, you may not use, modify, copy, publish, distribute, disclose or
# transmit this software or the related documents without Intel's prior
# written permission.
#
# This software and the related documents are provided as is, with no
# express or implied warranties, other than those that are expressly
# stated in the License.
#

set(TARGET_NAME KmbUnitTests)

file(GLOB TEST_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${TARGET_NAME} ${TEST_SOURCES})

target_compile_options(${TARGET_NAME}
    PRIVATE
        -Wall
        -Werror)

target_link_libraries(${TARGET_NAME} PRIVATE IE::gtest IE::gtest_main kmbPlugin_test_static)

add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
