# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_LTO)
    ie_enable_lto()
endif()

set(TARGET_NAME "vpuxUnitTests")
set(EXCLUDED_UNIT_TESTS_DIR "")
set(OPTIONAL_UNIT_TESTS_INCLUDES "")
set(OPTIONAL_UNIT_TESTS_LIBS "")

#
# Add optional HDDL2 Tests
#

if(ENABLE_HDDL2_TESTS AND BUILD_SHARED_LIBS)
    list(APPEND OPTIONAL_UNIT_TESTS_INCLUDES
            "hddl2_tests/helpers"
            "${IE_MAIN_KMB_PLUGIN_TEST_DIR}/common_helpers"
            $<TARGET_PROPERTY:vpux_hddl2_backend_static,INTERFACE_INCLUDE_DIRECTORIES>
    )

    list(APPEND OPTIONAL_UNIT_TESTS_LIBS
        vpux_hddl2_backend_static
    )
else()
    list(APPEND EXCLUDED_UNIT_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/hddl2_tests")
endif()

#
# Add optional MCM tests
#

if(ENABLE_MCM_COMPILER)
    list(APPEND OPTIONAL_UNIT_TESTS_INCLUDES
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/mcmCompiler/src/"
    )

    list(APPEND OPTIONAL_UNIT_TESTS_LIBS
        vpux_mcm_frontend_static
    )
else()
    list(APPEND EXCLUDED_UNIT_TESTS_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/mcmCompiler"
        "${CMAKE_CURRENT_SOURCE_DIR}/frontend_mcm"
    )
endif()

set(VPUAL_BACKEND_INCLUDES_DIR
    "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/vpual_backend/")

add_subdirectory(kmb/test_utils)

if (AARCH64)
    get_target_property(NNINC NN INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(RemoteFlicINC RemoteFlic INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(DispatcherINC VpualDispatcher INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(XLinkINC XLink INTERFACE_INCLUDE_DIRECTORIES)
    string(CONCAT VPUAL_INCLUDES ${NNINC} " " ${RemoteFlicINC} " " ${DispatcherINC} " " ${XLinkINC})
endif()

addIeTargetTest(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    EXCLUDED_SOURCE_PATHS
        ${EXCLUDED_UNIT_TESTS_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/kmb/test_utils
    INCLUDES
        ${OPTIONAL_UNIT_TESTS_INCLUDES}
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include"
        ${VPUAL_INCLUDES}
        ${VPUAL_BACKEND_INCLUDES_DIR}
    LINK_LIBRARIES
        ${OPTIONAL_UNIT_TESTS_LIBS}
        vpux_elf
        IE::commonTestUtils
        IE::ngraph
        IE::gmock
        ${VPUMGR_LIBRARY}
        common_tests_helpers
        ${VPUX_PLUGIN_STATIC_TARGET}
        kmb_utils
        vpux_al
        vpux_utils
        vpux_mlir_compiler_static
        vpux_translate_utils_static
    LINK_LIBRARIES_WHOLE_ARCHIVE
        kmbTestData
    DEFINES
        DATA_PATH=\"${DATA_PATH}\"
        MODELS_PATH=\"${MODELS_PATH}\"
    LABELS
        KMB
)

set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "tests")
add_dependencies(${TARGET_NAME} throw_test_backend vpu3700_test_backend no_devices_test_backend)

if(AARCH64)
    target_link_libraries(${TARGET_NAME} PRIVATE vpux_vpual_backend_static)
endif()

enable_warnings_as_errors(${TARGET_NAME})
