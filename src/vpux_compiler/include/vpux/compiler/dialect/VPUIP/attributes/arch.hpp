//
// Copyright 2020 Intel Corporation.
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
#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"

#include <mlir/IR/BuiltinOps.h>

namespace vpux {
namespace VPUIP {

constexpr uint32_t HW_TIMER_ABSOLUTE_ADDR = 0x208200BC;

void setArch(mlir::ModuleOp module, ArchKind kind, Optional<int> numOfDPUGroups = None);
ArchKind getArch(mlir::ModuleOp module);

double getMemoryDerateFactor(IERT::MemoryResourceOp mem);
uint32_t getMemoryBandwidth(IERT::MemoryResourceOp mem);
double getProcessorFrequency(IERT::ExecutorResourceOp res);

StringLiteral getProcessorFrequencyAttrName();
StringLiteral getBandwidthAttrName();

}  // namespace VPUIP
}  // namespace vpux
