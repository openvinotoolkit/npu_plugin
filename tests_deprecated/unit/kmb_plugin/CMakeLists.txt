#
# Copyright 2019-2020 Intel Corporation.
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

addIeTargetTest(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    INCLUDES
        "${IE_MAIN_KMB_PLUGIN_TEST_DIR}/common_helpers"
        "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include"
    LINK_LIBRARIES
        IE::IESharedTests
        IE::vpu_common_lib
        IE::ngraph
        ${VPUSMM_LIB}
        IE::gtest
        IE::gmock
        IE::gtest_main
        kmb_utils
        kmbPlugin_test_static
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

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${TARGET_NAME} FOR_TARGETS ${TARGET_NAME} ALL)
endif()
