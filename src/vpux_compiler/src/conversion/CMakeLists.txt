#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
set(TARGET_NAME "npu_compiler_conversion")
add_mlir_library(${TARGET_NAME}
    STATIC ${SRC_FILES}
    EXCLUDE_FROM_LIBMLIR
    DISABLE_INSTALL
    LINK_LIBS
    PRIVATE
        npu_compiler_core
        npu_compiler_utils
        npu_utils
        npu_compiler_dialect_vpumi37xx
        npu_compiler_dialect_elf
    PUBLIC
        mlir-dependencies
    DEPENDS
        MLIRVPUXIncGenList)
target_include_directories(${TARGET_NAME}
    SYSTEM PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/sw_runtime_kernels>
        ${VPU_COMPILER_BIN_INCLUDE_DIR}
        $<BUILD_INTERFACE:${MLIR_INCLUDE_DIRS}>)
target_include_directories(${TARGET_NAME}
    PRIVATE
        ${VPU_COMPILER_SRC_INCLUDE_DIR})
add_src_target($<BUILD_INTERFACE:${TARGET_NAME}>)
vpux_enable_clang_format(${TARGET_NAME})
enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
