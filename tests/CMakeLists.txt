# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(libs)

add_subdirectory(unit)

if(ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(functional)
endif()

vpux_setup_lit_tool(flatc)
add_subdirectory(lit)
