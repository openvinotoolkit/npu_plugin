#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
set(TARGET_NAME "npu_compiler_dialect_vpuip")
add_mlir_dialect_library(${TARGET_NAME}
    STATIC
        ${SRC_FILES}
        # FIXME: Need a solution to replace the function (E#88484)
        ${PROJECT_SOURCE_DIR}/src/vpux_compiler/src/dialect/VPU/utils/ppe_utils.cpp
        ${PROJECT_SOURCE_DIR}/src/vpux_compiler/src/dialect/VPU/utils/tile_utils.cpp
        ${PROJECT_SOURCE_DIR}/src/vpux_compiler/src/dialect/IE/utils/deconvolution_utils.cpp
    EXCLUDE_FROM_LIBMLIR
    DISABLE_INSTALL
    LINK_LIBS PRIVATE
        npu_utils
        npu_compiler_utils
        npu_compiler_core
        npu_compiler_act_kernels
        npu_al
    DEPENDS
        MLIRVPUXIncGenList)
target_include_directories(${TARGET_NAME}
    SYSTEM PRIVATE
        $<BUILD_INTERFACE:${MLIR_INCLUDE_DIRS}
        ${VPUNN_INCLUDE_DIRS}
        ${VPU_COMPILER_BIN_INCLUDE_DIR})
target_include_directories(${TARGET_NAME}
    PRIVATE
        ${VPU_COMPILER_SRC_INCLUDE_DIR})
add_src_target($<BUILD_INTERFACE:${TARGET_NAME}>)
vpux_enable_clang_format(${TARGET_NAME})
enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
