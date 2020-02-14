# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(options)

if((ARM OR AARCH64) AND (NOT DEFINED MCM_COMPILER_EXPORT_FILE OR NOT EXISTS ${MCM_COMPILER_EXPORT_FILE}))
    ie_option(ENABLE_MCM_COMPILER "Enable MCM compiler build" OFF)
else()
    ie_option(ENABLE_MCM_COMPILER "Enable MCM compiler build" ON)
endif()

if(ARM OR AARCH64)
    ie_option(ENABLE_HDDL2 "Enable HDDL2 Plugin" OFF)
else()
    # TODO: enable by default when linker issues is fixed
    ie_option(ENABLE_HDDL2 "Enable HDDL2 Plugin" OFF)
endif()

# TODO: Switch to off when CI will use real device
ie_option(ENABLE_HDDL2_SIMULATOR "Enable XLink emulator for HDDL2 Plugin" ON)

ie_option(ENABLE_VPUAL "Enable VPUAL" ON)
ie_option(ENABLE_VPUAL_MODEL "Enable VPUAL model" OFF)

ie_option(ENABLE_KMB_SAMPLES "Enable KMB samples" ON)
