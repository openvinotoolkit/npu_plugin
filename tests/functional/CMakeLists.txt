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
            ${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/tests/functional/shared_tests_instances
        LINK_LIBRARIES
            IE::funcSharedTests
            IE::vpu_common_lib
        ADD_CPPLINT
        LABELS
            KMB
)
