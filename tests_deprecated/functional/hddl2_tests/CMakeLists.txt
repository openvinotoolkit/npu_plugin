#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

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
        IE::funcTestUtils
        vpux_utils
        kmb_utils
        vpux_mcm_frontend_static
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

add_dependencies(${TARGET_NAME} ${VPUX_PLUGIN_TARGET} vpux_hddl2_backend)
enable_warnings_as_errors(${TARGET_NAME})

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT ${VPUX_TESTS_COMPONENT}
        EXCLUDE_FROM_ALL
)

# TODO: Remove duplication EISW-31024
install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL
)
