# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#disable_deprecated_warnings()

if (ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(functional)
endif()

if (ENABLE_BEH_TESTS)
    disable_deprecated_warnings()
    add_subdirectory(behavior)
endif()

add_subdirectory(unit)
