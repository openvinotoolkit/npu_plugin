//
// Copyright (C) 2022 Intel Corporation.
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
#include "vpux/compiler/dialect/VPUIPRegMapped/attributes/enums.hpp"

#include <mlir/IR/BuiltinOps.h>

namespace vpux {
namespace VPUIPRegMapped {

constexpr uint32_t HW_TIMER_ABSOLUTE_ADDR = 0x208200BC;

ArchKind getArch(mlir::ModuleOp module);

double getMemoryDerateFactor(IE::MemoryResourceOp mem);
uint32_t getMemoryBandwidth(IE::MemoryResourceOp mem);

}  // namespace VPUIPRegMapped
}  // namespace vpux
