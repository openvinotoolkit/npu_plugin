# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (ENABLE_HDDL2_TESTS)
    add_subdirectory(hddl2_tests)
endif()

if (mcmCompiler_FOUND)
    add_subdirectory(frontend_mcm)
endif()

# TODO: It is disabled for Windows due dependency on Boost.fs
if (UNIX)
    set(TARGET_NAME "KmbFunctionalTests")

    find_package(Boost REQUIRED COMPONENTS system filesystem)

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
            DEPENDENCIES
                kmb_custom_kernels
            LINK_LIBRARIES
                IE::IESharedTests
                IE::vpu_common_lib
                IE::ngraph
                ${VPUSMM_LIB}
                kmb_utils
                Boost::filesystem
            LINK_LIBRARIES_WHOLE_ARCHIVE
                kmbTestData
            LABELS  # Must be the last parameter. Reason: see description comment for addIeTargetTest().
                KMB
                VPU
    )

    add_dependencies(${TARGET_NAME} kmbPlugin)
    if ((X86 OR X86_64) AND UNIX)
        add_dependencies(${TARGET_NAME} VPUXPlugin)
        if(ENABLE_HDDL2)
            add_dependencies(${TARGET_NAME} HDDL2Plugin)
        endif()
    endif()

        target_compile_options(${TARGET_NAME}
            PRIVATE
                -Wall
                -Werror
                -Wextra)
endif()
