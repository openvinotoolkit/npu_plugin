# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include (options)

# Enable MCM compiler by default
if(ARM OR AARCH64)
	ie_option(ENABLE_MCM_COMPILER "Enable mcm compiler" OFF)
else()
	ie_option(ENABLE_MCM_COMPILER "Enable mcm compiler" ON)
endif()

ie_option(ENABLE_VPUAL "Enable VPUAL" ON)
