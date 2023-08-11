//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/utils/asm.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpImplementation.h>

using namespace vpux;

bool IE::isActShaveKernel(mlir::Operation* operation) {
    const auto module = operation->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    if (arch != VPU::ArchKind::VPUX37XX) {
        return false;
    }

    return VPUIP::NCEInvariant::verifyKernel(operation, Logger::global()).failed() &&
           operation->hasTrait<IE::EltwiseOp>();
}

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IE/generated/ops.cpp.inc>
