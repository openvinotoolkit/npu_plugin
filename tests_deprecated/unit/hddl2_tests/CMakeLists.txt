#
# Copyright 2020 Intel Corporation.
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

set(TARGET_NAME HDDL2UnitTests)

addIeTargetTest(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    INCLUDES
        "helpers"
        "${IE_MAIN_KMB_PLUGIN_TEST_DIR}/common_helpers"
    LINK_LIBRARIES
        IE::IESharedTests
        HDDL2Plugin_test_static
        HddlUnite
        ${XLINK_LIB}
        IE::gtest
        IE::gtest_main
        frontend_mcm
    LINK_LIBRARIES_WHOLE_ARCHIVE
        kmbTestData
)

if (UNIX)
    target_compile_options(${TARGET_NAME}
        PRIVATE
            -Wall
            -Werror
            -Wextra)
endif()
