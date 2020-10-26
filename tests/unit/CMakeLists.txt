# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "kmbUnitTests")

addIeTargetTest(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    INCLUDES
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include"
    LINK_LIBRARIES
        IE::commonTestUtils
        IE::ngraph
        IE::gmock
        ${VPUSMM_LIB}
        common_tests_helpers
        ${KMB_PLUGIN_STATIC_TARGET}
        kmb_utils
        vpux_al
        frontend_mcm_static
        vpual_backend_static

    DEFINES
        DATA_PATH=\"${DATA_PATH}\"
        MODELS_PATH=\"${MODELS_PATH}\"
    LABELS
        KMB
)

enable_warnings_as_errors(${TARGET_NAME})
