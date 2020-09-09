# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME KmbBehaviorTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include
        DEFINES
            INSTANTIATE_TESTS=1
        LINK_LIBRARIES
            IE::IEBehaviorSharedTests
            IE::ieTestHelpers
        LINK_LIBRARIES_WHOLE_ARCHIVE
            kmbTestData
        LABELS  # Must be the last parameter. Reason: see description comment for addIeTargetTest().
            KMB
)

enable_warnings_as_errors(${TARGET_NAME})
