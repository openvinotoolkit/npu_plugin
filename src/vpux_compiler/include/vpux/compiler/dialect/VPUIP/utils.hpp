//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>

namespace vpux {
namespace VPUIP {

//
// Profiling
//

constexpr uint32_t HW_TIMER_ABSOLUTE_ADDR = 0x208200BC;
// DMA Profiling
constexpr uint32_t HW_DMA_PROFILING_MAX_BUFFER_SIZE = 256;
// DPU Profiling consist of 2 64bit timestamps(start and stop)
constexpr uint16_t HW_DPU_PROFILING_SIZE_BYTES = 16;
constexpr uint32_t HW_DPU_PROFILING_MAX_BUFFER_SIZE = 128;
// UPA Profiling consist of 2 64bit timestamps(start and stop) + 2 32bit for active and stall counters
constexpr uint16_t HW_UPA_PROFILING_SIZE_BYTES = 24;

//
// Run-time info
//

double getMemoryDerateFactor(IERT::MemoryResourceOp mem);
uint32_t getMemoryBandwidth(IERT::MemoryResourceOp mem);
double getProcessorFrequency(IERT::ExecutorResourceOp res);

//
// MemoryLocation utility
//

VPU::MemoryKind getMemoryKind(MemoryLocation location);
MemoryLocation getMemoryLocation(VPU::MemoryKind memKind);

bool isMemoryCompatible(MemoryLocation location, mlir::MemRefType memref);

//
// DW Convolution utility
//

mlir::Value alignDepthWiseWeightsTensor(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter);

}  // namespace VPUIP
}  // namespace vpux
