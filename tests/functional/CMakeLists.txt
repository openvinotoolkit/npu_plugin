#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

set(TARGET_NAME npuFuncTests)

addIeTargetTest(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/vpux_al/include"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/tests/functional/shared_tests_instances"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/tests/functional/behavior"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/tests/functional/subgraph_tests"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2"
        "${CMAKE_CURRENT_SOURCE_DIR}/internal"
        "${CMAKE_CURRENT_SOURCE_DIR}/shared_test_classes/include"
        "${PROJECT_BINARY_DIR}/src/vpux_compiler/include"
        ${LLVM_INCLUDE_DIRS}
    DEPENDENCIES
        openvino_intel_npu_plugin
    LINK_LIBRARIES
        openvino::funcSharedTests
        openvino::format_reader
        openvino::reference
        openvino::runtime
        npu_model_utils
        common_tests_helpers
        npu_llvm_utils
        npu_ov_utils
        npu_profiling_utils
        npu_plugin_utils
        npu_al
    DEFINES
        DATA_PATH=\"${DATA_PATH}\"
        MODELS_PATH=\"${MODELS_PATH}\"
    LABELS
        KMB
)

set_target_properties(${TARGET_NAME} PROPERTIES
                      FOLDER ${CMAKE_CURRENT_SOURCE_DIR}
                      CXX_STANDARD 17)

if(MSVC)
    # Enforce standards conformance on MSVC
    target_compile_options(${TARGET_NAME}
        PRIVATE
            /Zc:preprocessor
    )
endif()

vpux_enable_clang_format(${TARGET_NAME})

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT ${VPUX_TESTS_COMPONENT}
        EXCLUDE_FROM_ALL
)

# TODO: Remove duplication E#31024
install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL
)

if(ENABLE_DEVELOPER_BUILD AND UNIX)
    # add npuFuncTest improved with gtest-parallel to ctest
    # we are using run_gtest_parallel_manual_threads.sh script to provide
    # PARALLEL_WORKERS env variable as -w workers command line argument
    add_test(NAME ${TARGET_NAME}-parallel
            COMMAND
                ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/scripts/run_gtest_parallel_manual_threads.sh
                ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/gtest-parallel/gtest-parallel
                $<TARGET_FILE_DIR:${TARGET_NAME}>/${TARGET_NAME}
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH} )
            set_tests_properties(${TARGET_NAME}-parallel PROPERTIES
            LABELS "NPU;PlatformSpecific;3720"
    )
endif()
