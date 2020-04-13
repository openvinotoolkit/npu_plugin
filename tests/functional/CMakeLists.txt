# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (ENABLE_HDDL2 AND ENABLE_HDDL2_TESTS)
    add_subdirectory(hddl2_tests)
endif()

if (ENABLE_MCM_COMPILER)
    add_subdirectory(frontend_mcm)
endif()

set(TARGET_NAME "KmbFunctionalTests")

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT
            ${CMAKE_CURRENT_SOURCE_DIR}/kmb_tests
        ADDITIONAL_SOURCE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/base
            ${CMAKE_CURRENT_SOURCE_DIR}/common
            ${CMAKE_CURRENT_SOURCE_DIR}/new
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}/../common_helpers
            ${CMAKE_CURRENT_SOURCE_DIR}/new
            ${CMAKE_CURRENT_SOURCE_DIR}/base
            ${CMAKE_CURRENT_SOURCE_DIR}/common
            ${CMAKE_CURRENT_SOURCE_DIR}/regression_tests/helpers
            ${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include
            ${IE_MAIN_SOURCE_DIR}/src/vpu/graph_transformer/include
        DEFINES
            -DKMB_ALPHA_TESTS_DATA_PATH="${MODELS_PATH}"
        LINK_LIBRARIES
            IE::IESharedTests
            IE::vpu_common_lib
            IE::ngraph
            ${VPUSMM_LIB}
            kmb_utils
            ieModelsHelpersKmb
        ADD_CPPLINT
        LABELS  # Must be the last parameter. Reason: see description comment for addIeTargetTest().
            KMB
            VPU
)

target_compile_options(${TARGET_NAME}
    PRIVATE
        -Wall
        -Werror)
