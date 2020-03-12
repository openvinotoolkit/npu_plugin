# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(options)

ie_dependent_option(ENABLE_MCM_COMPILER "Enable mcmCompiler" ON "NOT CMAKE_CROSSCOMPILING OR DEFINED MCM_COMPILER_EXPORT_FILE" OFF)

ie_dependent_option (ENABLE_VPUAL "Enable VPUAL" ON "ARM OR AARCH64" OFF)

# TODO: enable by default when linker issues is fixed
ie_dependent_option (ENABLE_HDDL2 "Enable HDDL2 Plugin" OFF "NOT ARM;NOT AARCH64" OFF)

# TODO: Switch to off when CI will use real device
ie_option(ENABLE_HDDL2_SIMULATOR "Enable XLink emulator for HDDL2 Plugin" ON)

ie_option(ENABLE_VPUAL_MODEL "Enable VPUAL model" OFF)

ie_option(ENABLE_KMB_SAMPLES "Enable KMB samples" ON)

ie_option(ENABLE_EXPORT_SYMBOLS "Enable compiler -fvisibility=default and linker -export-dynamic options" OFF)
