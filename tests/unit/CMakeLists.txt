# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "vpuxUnitTests")

if(ENABLE_HDDL2_TESTS)
    set(EXCLUDED_HDDL2_UNIT_TESTS_DIR "")
    set(HDDL2_INCLUDES
            "hddl2_tests/helpers"
            "${IE_MAIN_KMB_PLUGIN_TEST_DIR}/common_helpers"
        )

    set(HDDL2_LIBS
            IE::gtest
            IE::gtest_main
            IE::IESharedTests
            HddlUnite
            hddl2_backend_static
            ${XLINK_LIB}
        )
else()
    set(EXCLUDED_HDDL2_UNIT_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/hddl2_tests")
endif()

set(MCM_SCHEDULER_INCLUDES_DIR
    "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/mcmCompiler/src/")

addIeTargetTest(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    EXCLUDED_SOURCE_PATHS ${EXCLUDED_HDDL2_UNIT_TESTS_DIR}
    INCLUDES
        ${HDDL2_INCLUDES}
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include"
        ${MCM_SCHEDULER_INCLUDES_DIR}
    LINK_LIBRARIES
        ${HDDL2_LIBS}
        IE::commonTestUtils
        IE::ngraph
        IE::gmock
        ${VPUMGR_LIBRARY}
        common_tests_helpers
        ${VPUX_PLUGIN_STATIC_TARGET}
        kmb_utils
        vpux_al
        vpux_utils
        vpux_compiler_static
        frontend_mcm_static
        vpual_backend_static
    LINK_LIBRARIES_WHOLE_ARCHIVE
        kmbTestData

    DEFINES
        DATA_PATH=\"${DATA_PATH}\"
        MODELS_PATH=\"${MODELS_PATH}\"
    LABELS
        VPUX
)

set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "tests")

enable_warnings_as_errors(${TARGET_NAME})
