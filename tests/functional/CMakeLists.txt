# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME vpuxFuncTests)

addIeTargetTest(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/vpux_al/include"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/tests/functional/shared_tests_instances"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/tests/functional/behavior"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/tests/functional/subgraph_tests"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/custom_extension_library"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/vpual_backend"
        "${CMAKE_CURRENT_SOURCE_DIR}/internal"
        "${CMAKE_CURRENT_SOURCE_DIR}/shared_test_classes/include"
        ${LLVM_INCLUDE_DIRS}
    DEPENDENCIES
        openvino_intel_vpux_plugin
    LINK_LIBRARIES
        IE::funcSharedTests
        IE::ngraph
        IE::ngraph_reference
        IE::format_reader
        IE::inference_engine
        kmb_utils
        kmb_custom_ocl_kernels
        kmb_custom_cpp_kernels
        common_tests_helpers
        kmb_custom_extension
        custom_extension_library
        vpux_utils
    DEFINES
        DATA_PATH=\"${DATA_PATH}\"
        MODELS_PATH=\"${MODELS_PATH}\"
    LABELS
        KMB
)

if(MSVC)
    # Enforce standards conformance on MSVC
    target_compile_options(${TARGET_NAME}
        PRIVATE
            /Zc:preprocessor
    )
endif()

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
