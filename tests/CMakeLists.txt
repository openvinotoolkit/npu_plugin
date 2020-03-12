# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(unit/kmb_plugin)

if(ENABLE_HDDL2)
    add_subdirectory(unit/hddl2_tests)
endif()

if (ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(functional)
endif()

if (ENABLE_BEH_TESTS)
    add_subdirectory(behavior)
endif()

if (ARM OR AARCH64)
    add_subdirectory(gapi_preproc)
endif()
