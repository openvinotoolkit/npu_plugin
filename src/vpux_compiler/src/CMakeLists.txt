#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

# Set BUILD_SHARED_LIBS=OFF explicitly to ensure files are built in static lib
# See function add_mlir_library in AddMLIR for the details
set(BUILD_SHARED_LIBS OFF)
set(VPU_COMPILER_SRC_INCLUDE_DIR
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src/vpux_compiler/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/vpux_compiler/include>)
set(VPU_COMPILER_BIN_INCLUDE_DIR
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src/vpux_compiler/tblgen>)

add_subdirectory(act-kernels)
add_subdirectory(conversion)
add_subdirectory(core)
add_subdirectory(dialect/const)
add_subdirectory(dialect/ELF)
add_subdirectory(dialect/EMU)
add_subdirectory(dialect/IE)
add_subdirectory(dialect/IERT)
add_subdirectory(dialect/VPU)
add_subdirectory(dialect/VPU37XX)
add_subdirectory(dialect/VPUIP)
add_subdirectory(dialect/VPUMI37XX)
add_subdirectory(dialect/VPURegMapped)
add_subdirectory(dialect/VPURT)
add_subdirectory(frontend)
add_subdirectory(tools)
add_subdirectory(utils)
add_subdirectory(VPU30XX)
add_subdirectory(VPU37XX)
