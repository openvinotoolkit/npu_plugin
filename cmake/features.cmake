# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(options)

ie_dependent_option(ENABLE_KMB_SAMPLES "Enable KMB samples" ON "AARCH64" OFF)

ie_dependent_option(ENABLE_HDDL2 "Enable HDDL2 Plugin" ON "NOT ARM;NOT AARCH64" OFF)
ie_dependent_option(ENABLE_HDDL2_TESTS "Enable Unit and Functional tests for HDDL2 Plugin" OFF "ENABLE_HDDL2;ENABLE_TESTS" OFF)

ie_option(ENABLE_TESTS "Enable KMB tests" ON)
ie_dependent_option(ENABLE_MODELS "download all models required for functional testing" ON "ENABLE_FUNCTIONAL_TESTS" OFF)
ie_dependent_option(ENABLE_VALIDATION_SET "download validation_set required for functional testing" ON "ENABLE_FUNCTIONAL_TESTS" OFF)

ie_option(ENABLE_EXPORT_SYMBOLS "Enable compiler -fvisibility=default and linker -export-dynamic options" OFF)
ie_option(ENABLE_M2I "Enable Media-to-Inference (M2I) module for image pre-processing" OFF)

# TODO: the option works only for x86 unix platform now
ie_dependent_option(ENABLE_MCM_FROM_REPO "Enable compiler to be built from sources" OFF "NOT ARM; NOT AARCH64; NOT WIN32" OFF)

function (print_enabled_kmb_features)
    message(STATUS "KmbPlugin enabled features: ")
    message(STATUS "")
    foreach(_var ${IE_OPTIONS})
        message(STATUS "    ${_var} = ${${_var}}")
    endforeach()
    message(STATUS "")
endfunction()

