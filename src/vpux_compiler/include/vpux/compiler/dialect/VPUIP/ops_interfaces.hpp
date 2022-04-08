//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_writer.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace vpux {
namespace VPUIP {

class BlobWriter;

//
// TaskOpInterface
//

using MemoryEffect = mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>;
void getTaskEffects(mlir::Operation* op, SmallVectorImpl<MemoryEffect>& effects);

IndexedSymbolAttr getExecutorAttr(mlir::Operation* op, VPU::ExecutorKind kind);

IndexedSymbolAttr getTaskOpExecutor(mlir::Operation* op);

//
// UPATask
//

mlir::LogicalResult verifyUPATask(mlir::Operation* op);

template <typename ConcreteOp>
class UPATask : public mlir::OpTrait::TraitBase<ConcreteOp, UPATask> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifyUPATask(op);
    }

    static VPU::ExecutorKind getExecutorKind() {
        return VPU::ExecutorKind::SHAVE_UPA;
    }
};

//
// Legacy4D
//

mlir::LogicalResult verifyLegacy4D(mlir::Operation* op);

template <typename ConcreteOp>
class Legacy4D : public mlir::OpTrait::TraitBase<ConcreteOp, Legacy4D> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifyLegacy4D(op);
    }
};

}  // namespace VPUIP
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/VPUIP/generated/ops_interfaces.hpp.inc>
