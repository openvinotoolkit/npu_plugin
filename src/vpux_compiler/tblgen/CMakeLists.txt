#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

add_subdirectory(vpux/compiler/core)
add_subdirectory(vpux/compiler/conversion)
add_subdirectory(vpux/compiler/dialect/const)
add_subdirectory(vpux/compiler/dialect/IE)
add_subdirectory(vpux/compiler/dialect/VPU)
add_subdirectory(vpux/compiler/VPU30XX/conversion)
add_subdirectory(vpux/compiler/VPU30XX/dialect/IE)
add_subdirectory(vpux/compiler/VPU30XX/dialect/VPUIP)
add_subdirectory(vpux/compiler/VPU37XX/conversion)
add_subdirectory(vpux/compiler/VPU37XX/dialect/IE)
add_subdirectory(vpux/compiler/VPU37XX/dialect/VPU)
add_subdirectory(vpux/compiler/VPU37XX/dialect/VPUIP)
add_subdirectory(vpux/compiler/dialect/EMU)
add_subdirectory(vpux/compiler/dialect/IERT)
add_subdirectory(vpux/compiler/dialect/VPUIP)
add_subdirectory(vpux/compiler/dialect/VPURT)
add_subdirectory(vpux/compiler/dialect/ELF)
add_subdirectory(vpux/compiler/dialect/VPUMI37XX)
add_subdirectory(vpux/compiler/dialect/VPURegMapped)
add_subdirectory(vpux/compiler/dialect/VPU37XX)
