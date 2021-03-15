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

set(TARGET_NAME "HDDL2FunctionalTests")

file(GLOB_RECURSE HDDL2_TESTS_SOURCES "*.cpp")

find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)

addIeTargetTest(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    INCLUDES
        "helpers"
        "cases"
        "${IE_MAIN_KMB_PLUGIN_TEST_DIR}/common_helpers"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include/hddl2"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include/vpu"
    LINK_LIBRARIES
        IE::IESharedTests
        IE::vpu_common_lib
        kmb_utils
        frontend_mcm_static
        common_tests_helpers
    LINK_LIBRARIES_WHOLE_ARCHIVE
        kmbTestData
    LABELS
        VPUX
)

if(OpenCV_FOUND)
    target_compile_definitions(${TARGET_NAME} PRIVATE USE_OPENCV)
    target_link_libraries(${TARGET_NAME} PRIVATE opencv_core opencv_imgproc opencv_imgcodecs)
endif()

add_dependencies(${TARGET_NAME} ${VPUX_PLUGIN_TARGET} hddl2_backend)
enable_warnings_as_errors(${TARGET_NAME})
