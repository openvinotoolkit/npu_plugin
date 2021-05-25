#
# Copyright 2019-2020 Intel Corporation.
#
# LEGAL NOTICE: Your use of this software and any required dependent software
# (the "Software Package") is subject to the terms and conditions of
# the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
# which may also include notices, disclaimers, or license terms for
# third party or open source software included in or with the Software Package,
# and your use indicates your acceptance of all such terms. Please refer
# to the "third-party-programs.txt" or other similarly-named text file
# included with the Software Package for additional details.
#

set(IE_MAIN_KMB_PLUGIN_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(models_helpers)

if (ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(functional)
endif()

if(AARCH64)
    add_subdirectory(gapi_preproc)
endif()
