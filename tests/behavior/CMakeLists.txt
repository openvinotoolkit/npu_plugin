# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME KmbBehaviorTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/plugin_tests
            ${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include
        DEFINES
            INSTANTIATE_TESTS=1
        LINK_LIBRARIES
            IE::IEBehaviorSharedTests
            IE::ieTestHelpers
            ieModelsHelpersKmb
        ADD_CPPLINT
        LABELS  # Must be the last parameter. Reason: see description comment for addIeTargetTest().
            KMB
)

target_compile_options(${TARGET_NAME}
        PRIVATE
        -Wall
        -Werror
        )
