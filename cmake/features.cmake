# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(options)

ie_dependent_option(ENABLE_HDDL2 "Enable HDDL2 Plugin" ON "NOT ARM;NOT AARCH64" OFF)
ie_dependent_option(ENABLE_HDDL2_TESTS "Enable Unit and Functional tests for HDDL2 Plugin" OFF "ENABLE_HDDL2;ENABLE_TESTS" OFF)

ie_dependent_option(ENABLE_KMB_SAMPLES "Enable KMB samples" ON "AARCH64" OFF)

ie_dependent_option(ENABLE_MODELS "download all models required for functional testing" ON "ENABLE_FUNCTIONAL_TESTS" OFF)
ie_dependent_option(ENABLE_VALIDATION_SET "download validation_set required for functional testing" ON "ENABLE_FUNCTIONAL_TESTS" OFF)

ie_option(ENABLE_EXPORT_SYMBOLS "Enable compiler -fvisibility=default and linker -export-dynamic options" OFF)
