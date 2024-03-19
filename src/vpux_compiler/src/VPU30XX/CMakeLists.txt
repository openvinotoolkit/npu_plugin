#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
set(TARGET_NAME "npu_compiler_vpu30xx")
add_mlir_library(${TARGET_NAME}
    STATIC ${SRC_FILES}
    EXCLUDE_FROM_LIBMLIR
    DISABLE_INSTALL
    LINK_LIBS PRIVATE
        npu_compiler_conversion
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpurt
        npu_compiler_core
        npu_compiler_dialect_const
        npu_compiler_dialect_ie
        npu_compiler_dialect_vpu_interfaces
        npu_compiler_dialect_vpu_transforms
        npu_llvm_utils
        npu_ov_utils
        flatbuffers
        npu_elf)
add_dependencies(${TARGET_NAME} MLIRVPUXIncGenList)

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
