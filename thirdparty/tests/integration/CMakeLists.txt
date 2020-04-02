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

file(GLOB TEST_INCLUDE
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

file(GLOB TEST_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

if(NOT DEFINED INTEGRATION_TESTS_ENABLE_IE)
    set(INTEGRATION_TESTS_ENABLE_IE ON)
endif()

function(addVpuTest TARGET_NAME PLUGIN_NAME)
    add_executable(${TARGET_NAME} ${TEST_INCLUDE} ${TEST_SRC})
    target_compile_options(${TARGET_NAME}
            PRIVATE
            -Wall
            -Werror
            -Wno-error=strict-aliasing
            -Wno-strict-aliasing
            -Wno-unused-variable)

    if (INTEGRATION_TESTS_ENABLE_IE)
        add_definitions(-DINTEGRATION_TESTS_ENABLE_IE)
        target_link_libraries(${TARGET_NAME} PRIVATE IE::gtest IE::gtest_main IE::ieTestHelpers NN vpusmm)

    else()
        find_package(GTest REQUIRED)
        if (NOT INTEGRATION_TESTS_BLOB_FILE OR NOT INTEGRATION_TESTS_INPUT_FILE)
            message(FATAL_ERROR "IE is disabled, but INTEGRATION_TESTS_BLOB_FILE or INTEGRATION_TESTS_INPUT_FILE are undefined")
        endif()
        add_definitions(
            -DINTEGRATION_TESTS_BLOB_FILE="${INTEGRATION_TESTS_BLOB_FILE}"
            -DINTEGRATION_TESTS_INPUT_FILE="${INTEGRATION_TESTS_INPUT_FILE}"
        )
        target_link_libraries(${TARGET_NAME} PRIVATE ${GTEST_LIBRARIES} NN vpusmm)
    endif()

    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
    set_property(TEST ${TARGET_NAME} PROPERTY LABELS VPU ${PLUGIN_NAME})
endfunction()

addVpuTest(KmbIntegrationTests kmbPlugin)
