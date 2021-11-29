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
