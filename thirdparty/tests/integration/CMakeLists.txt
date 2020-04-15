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

set(TARGET_NAME "KmbIntegrationTests")

if(NOT DEFINED INTEGRATION_TESTS_ENABLE_IE)
    set(INTEGRATION_TESTS_ENABLE_IE ON)
endif()

if (INTEGRATION_TESTS_ENABLE_IE)
    add_definitions(-DINTEGRATION_TESTS_ENABLE_IE)
    set(LINK_LIBRARIES IE::gtest IE::gtest_main NN vpusmm IE::ieTestHelpers)
else()
    find_package(GTest REQUIRED)

    if (NOT INTEGRATION_TESTS_BLOB_FILE OR NOT INTEGRATION_TESTS_INPUT_FILE)
        message(FATAL_ERROR "IE is disabled, but INTEGRATION_TESTS_BLOB_FILE or INTEGRATION_TESTS_INPUT_FILE are undefined")
    endif()

    add_definitions(
            -DINTEGRATION_TESTS_BLOB_FILE="${INTEGRATION_TESTS_BLOB_FILE}"
            -DINTEGRATION_TESTS_INPUT_FILE="${INTEGRATION_TESTS_INPUT_FILE}"
    )

    set(LINK_LIBRARIES ${GTEST_LIBRARIES} NN vpusmm)
endif()

addIeTargetTest(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    INCLUDES
        "helpers"
        "cases"
        "${IE_MAIN_KMB_PLUGIN_TEST_DIR}/common_helpers"
        "${IE_MAIN_SOURCE_DIR}/include"
        "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include/hddl2"
    LINK_LIBRARIES
        ${LINK_LIBRARIES}
    LINK_LIBRARIES_WHOLE_ARCHIVE
        kmbTestData
    LABELS
        VPU
        kmbPlugin
)

target_compile_options(${TARGET_NAME}
    PRIVATE
        -Wall
        -Werror
        -Wno-error=strict-aliasing
        -Wno-strict-aliasing
        -Wno-unused-variable)
