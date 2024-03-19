#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
set(TARGET_NAME "npu_compiler_frontend")
add_mlir_library(${TARGET_NAME}
    STATIC ${SRC_FILES}
    EXCLUDE_FROM_LIBMLIR
    DISABLE_INSTALL
    LINK_LIBS PRIVATE
        npu_compiler_core
        npu_compiler_utils
        npu_llvm_utils
        npu_ov_utils
        flatbuffers
        npu_elf)
add_dependencies(${TARGET_NAME} MLIRVPUXIncGenList)

target_include_directories(${TARGET_NAME}
    SYSTEM PRIVATE
        $<BUILD_INTERFACE:${MLIR_INCLUDE_DIRS}>
        ${VPU_COMPILER_BIN_INCLUDE_DIR})
target_include_directories(${TARGET_NAME}
    PRIVATE
        ${VPU_COMPILER_SRC_INCLUDE_DIR})
link_system_libraries(${TARGET_NAME}
    PUBLIC
        npu_ngraph_transformations)
add_src_target($<BUILD_INTERFACE:${TARGET_NAME}>)
vpux_enable_clang_format(${TARGET_NAME})
enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
