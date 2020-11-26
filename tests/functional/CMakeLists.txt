# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME kmbFuncTests)

addIeTargetTest(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/tests/functional/shared_tests_instances"
        "${CMAKE_CURRENT_SOURCE_DIR}/internal"
    DEPENDENCIES
        VPUXPlugin
    LINK_LIBRARIES
        IE::funcSharedTests
        IE::vpu_common_lib
        IE::ngraph
        IE::ngraph_reference
        IE::format_reader
        kmb_utils
        kmb_custom_kernels
    DEFINES
        DATA_PATH=\"${DATA_PATH}\"
        MODELS_PATH=\"${MODELS_PATH}\"
    LABELS
        KMB
)

if(ENABLE_EXPERIMENTAL_MLIR)
    target_compile_definitions(${TARGET_NAME}
        PRIVATE
            ENABLE_EXPERIMENTAL_MLIR
    )
endif()
