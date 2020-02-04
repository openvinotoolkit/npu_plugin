# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(unit)

if(ENABLE_HDDL2 AND ENABLE_HDDL2_UNIT_TESTS)
    add_subdirectory(unit/hddl2_tests)
endif()

if (ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(functional)
endif()

if (ENABLE_BEH_TESTS)
    add_subdirectory(behavior)
endif()

if (ENABLE_VPUAL)
    add_subdirectory(gapi_preproc)
endif()
