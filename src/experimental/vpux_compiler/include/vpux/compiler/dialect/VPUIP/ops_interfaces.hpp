//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"
#include "vpux/compiler/dialect/VPUIP/blob_writer.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace vpux {
namespace VPUIP {

//
// Forward declarations
//

class BlobWriter;

//
// verifyUPATask
//

mlir::LogicalResult verifyUPATask(mlir::Operation* op);

//
// getTaskEffects
//

using MemoryEffect = mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>;

void getTaskEffects(mlir::Operation* op, SmallVectorImpl<MemoryEffect>& effects);

//
// SameShape
//

mlir::LogicalResult verifySameShape(mlir::Operation* op);

template <typename ConcreteOp>
class SameShape : public mlir::OpTrait::TraitBase<ConcreteOp, SameShape> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameShape(op);
    }
};

//
// SameElementType
//

mlir::LogicalResult verifySameElementType(mlir::Operation* op);

template <typename ConcreteOp>
class SameElementType : public mlir::OpTrait::TraitBase<ConcreteOp, SameElementType> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameElementType(op);
    }
};

//
// SameDimsOrder
//

mlir::LogicalResult verifySameDimsOrder(mlir::Operation* op);

template <typename ConcreteOp>
class SameDimsOrder : public mlir::OpTrait::TraitBase<ConcreteOp, SameDimsOrder> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameDimsOrder(op);
    }
};

//
// Legacy4D
//

mlir::LogicalResult verifyLegacy4D(mlir::Operation* op);

template <typename ConcreteOp>
class Legacy4D : public mlir::OpTrait::TraitBase<ConcreteOp, SameDimsOrder> {
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
