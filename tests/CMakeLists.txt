#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#

add_subdirectory(libs)

add_subdirectory(unit)

if(ENABLE_FUNCTIONAL_TESTS AND NOT BUILD_COMPILER_FOR_DRIVER)
    add_subdirectory(functional)
endif()

add_subdirectory(lit)

if (ENABLE_DIALECT_SHARED_LIBRARIES)
    add_subdirectory(smoke)
endif()
