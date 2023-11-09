#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
set(TARGET_NAME "npu_compiler_vpu37xx")
add_mlir_library(${TARGET_NAME}
    STATIC
    ${SRC_FILES}
    # FIXME: Need a solution to replace the function (E#88484)
    ${PROJECT_SOURCE_DIR}/src/vpux_compiler/src/core/act_profiling.cpp
    ${PROJECT_SOURCE_DIR}/src/vpux_compiler/src/core/async_deps_info.cpp
    ${PROJECT_SOURCE_DIR}/src/vpux_compiler/src/core/barrier_info.cpp
    ${PROJECT_SOURCE_DIR}/src/vpux_compiler/src/core/cost_model_utils.cpp
    ${PROJECT_SOURCE_DIR}/src/vpux_compiler/src/core/dpu_profiling.cpp
    ${PROJECT_SOURCE_DIR}/src/vpux_compiler/src/core/feasible_memory_scheduler_control_edges.cpp
    ${PROJECT_SOURCE_DIR}/src/vpux_compiler/src/core/feasible_memory_scheduler_spilling.cpp
    ${PROJECT_SOURCE_DIR}/src/vpux_compiler/src/core/feasible_memory_scheduler.cpp
    ${PROJECT_SOURCE_DIR}/src/vpux_compiler/src/core/linear_scan_handler.cpp
    ${PROJECT_SOURCE_DIR}/src/vpux_compiler/src/core/mem_live_range_info.cpp
    ${PROJECT_SOURCE_DIR}/src/vpux_compiler/src/core/overlap_DPU_and_DMA.cpp
    ${PROJECT_SOURCE_DIR}/src/vpux_compiler/src/core/schedule_analysis_utils.cpp
    ${PROJECT_SOURCE_DIR}/src/vpux_compiler/src/dialect/VPURT/passes/inference_execution_analysis.cpp
    EXCLUDE_FROM_LIBMLIR
    DISABLE_INSTALL
    LINK_LIBS PRIVATE
        npu_compiler_conversion
        npu_compiler_core
        npu_compiler_dialect_const
        npu_compiler_dialect_ie
        npu_compiler_dialect_vpu
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpurt
        npu_utils
    DEPENDS
        cpp_schema
        MLIRFuncOpsIncGen
        MLIRTransformsPassIncGen)

target_include_directories(${TARGET_NAME}
    SYSTEM PRIVATE
        ${VPUNN_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${MLIR_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/vpux_al/include>
        ${VPU_COMPILER_BIN_INCLUDE_DIR})
target_include_directories(${TARGET_NAME}
    PRIVATE
        ${VPU_COMPILER_SRC_INCLUDE_DIR})

add_src_target($<BUILD_INTERFACE:${TARGET_NAME}>)
vpux_enable_clang_format(${TARGET_NAME})
enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
