#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#

set(IE_MAIN_KMB_PLUGIN_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(models_helpers)

if (ENABLE_FUNCTIONAL_TESTS AND NOT BUILD_COMPILER_FOR_DRIVER)
    add_subdirectory(functional)
endif()
