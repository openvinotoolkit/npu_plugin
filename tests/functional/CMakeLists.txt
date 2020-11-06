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
        "${CMAKE_CURRENT_SOURCE_DIR}/internal"
    DEPENDENCIES
        VPUXPlugin
    LINK_LIBRARIES
        IE::funcTestUtils
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
