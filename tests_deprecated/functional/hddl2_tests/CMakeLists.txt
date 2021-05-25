#
# Copyright 2020 Intel Corporation.
#
# LEGAL NOTICE: Your use of this software and any required dependent software
# (the "Software Package") is subject to the terms and conditions of
# the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
# which may also include notices, disclaimers, or license terms for
# third party or open source software included in or with the Software Package,
# and your use indicates your acceptance of all such terms. Please refer
# to the "third-party-programs.txt" or other similarly-named text file
# included with the Software Package for additional details.
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
