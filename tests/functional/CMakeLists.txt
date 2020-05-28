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
            ${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include
        LINK_LIBRARIES
            IE::funcSharedTests
        ADD_CPPLINT
        LABELS
            KMB
)
