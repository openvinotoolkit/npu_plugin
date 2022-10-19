//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/preprocessing.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>

namespace vpux {
namespace VPUIP {

//
// Swizzling constants
//

// Swizzling key 5 requires bswizzled buffer to have start address aligned to 16k
constexpr uint64_t SWIZZLING_KEY_5 = 5;
// Swizzled buffers need to have space in CMX of size aligned to 512, because this
// it the smallest unit of CMX RAM cut
constexpr int64_t SWIZZLING_SIZE_ALIGNMENT = 512;

//
// Profiling
//

constexpr uint32_t HW_TIMER_ABSOLUTE_ADDR_30XX = 0x208200BC;
constexpr uint32_t HW_TIMER_ABSOLUTE_ADDR_37XX = 0x26029000;
// DMA Profiling consist of 2 32bit timestamps
constexpr uint16_t HW_DMA_PROFILING_SIZE_BYTES = 8;
constexpr uint32_t HW_DMA_PROFILING_MAX_BUFFER_SIZE = 256;
// DPU Profiling consist of 2 64bit timestamps(start and stop)
constexpr uint16_t HW_DPU_PROFILING_SIZE_BYTES_30XX = 16;
// DPU Profiling for 37XX use MODE0: // 8’h0, odu_tstamp[27:0], odu_wl_duration[27:0], {3’h0,sve_id[4:0]},
// idu_tstamp[27:0], idu_wl_duration[27:0]
constexpr uint16_t HW_DPU_PROFILING_SIZE_BYTES_37XX = 16;
constexpr uint16_t HW_DPU_PROFILING_SIZE_BYTES_40XX = 16;
constexpr uint32_t HW_DPU_PROFILING_MAX_BUFFER_SIZE =
        1024;  // Up to 64 DPU Tasks in single CMX DPU profiling buffer instance
// UPA Profiling consist of 2 64bit timestamps(start and stop) + 2 32bit for active and stall counters
constexpr uint16_t HW_UPA_PROFILING_SIZE_BYTES = 24;
// ActShave Profiling consist of 1 64bit start timestamp + 1 32bit duration + 1 32bit stall counter
constexpr uint16_t HW_ACT_SHAVE_PROFILING_SIZE_BYTES = 16;
constexpr uint32_t HW_ACT_SHAVE_PROFILING_MAX_BUFFER_SIZE = 128;
// 3720 architecture has 2 Mb CMX size. This define describes its half.
constexpr uint32_t HW_MIDDLE_OF_AVAILABLE_CMX_MEMORY_3720 = (1024 * 1024);

uint16_t getProfWorkloadSize(mlir::ModuleOp module);

//
// Run-time info
//

double getMemoryDerateFactor(IE::MemoryResourceOp mem);
uint32_t getMemoryBandwidth(IE::MemoryResourceOp mem);
double getProcessorFrequency(IE::ExecutorResourceOp res);
int64_t getNumAvailableBarriers(mlir::Operation* parentOp);

//
// DW Convolution utility
//

mlir::Value alignDepthWiseWeightsTensor(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter);

mlir::Value getTopBufferOfNCEClusterTiling(mlir::Operation* innerOp, mlir::Value buffer);

}  // namespace VPUIP
}  // namespace vpux
