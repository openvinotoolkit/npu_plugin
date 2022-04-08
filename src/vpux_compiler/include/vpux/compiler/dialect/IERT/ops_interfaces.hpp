//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"

#include "vpux/utils/core/small_string.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/DialectInterface.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace vpux {
namespace IERT {

//
// LayerOpInterface
//

mlir::LogicalResult verifyLayer(mlir::Operation* op);

mlir::OperandRange getLayerInputs(mlir::Operation* op);
mlir::OperandRange getLayerOutputs(mlir::Operation* op);

MutableArrayRef<mlir::OpOperand> getLayerInOpOperands(mlir::Operation* op);
MutableArrayRef<mlir::OpOperand> getLayerOutOpOperands(mlir::Operation* op);

mlir::Value getLayerViewSource(mlir::Operation* op, ptrdiff_t resultInd);

mlir::LogicalResult inferLayerReturnTypes(mlir::ValueRange operands, size_t numResults,
                                          SmallVectorImpl<mlir::Type>& inferredReturnTypes);

using MemoryEffect = mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>;
void getLayerEffects(mlir::Operation* op, SmallVectorImpl<MemoryEffect>& effects);

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

    static void inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info) {
        info.fill(info.getInput(0));
    }
};

//
// SameInOutDimsOrder
//

mlir::LogicalResult verifySameInOutDimsOrder(mlir::Operation* op);
void inferLayoutInfoSameInOutDimsOrder(IE::LayerLayoutInfo& info);

template <typename ConcreteOp>
class SameInOutDimsOrder : public mlir::OpTrait::TraitBase<ConcreteOp, SameInOutDimsOrder> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameInOutDimsOrder(op);
    }

    static void inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info) {
        inferLayoutInfoSameInOutDimsOrder(info);
    }
};

//
// SameInOutDimsOrder_NCHW_NHWC
//

mlir::LogicalResult verifySameInOutSpecificDimsOrder(mlir::Operation* op, ArrayRef<DimsOrder> supportedLayouts);
void inferLayoutInfoSameInOutSpecificDimsOrder(IE::LayerLayoutInfo& info, ArrayRef<DimsOrder> supportedLayouts);

template <typename ConcreteOp>
class SameInOutDimsOrder_NCHW_NHWC : public mlir::OpTrait::TraitBase<ConcreteOp, SameInOutDimsOrder_NCHW_NHWC> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameInOutSpecificDimsOrder(op, {DimsOrder::NCHW, DimsOrder::NHWC});
    }

    static void inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info) {
        inferLayoutInfoSameInOutSpecificDimsOrder(info, {DimsOrder::NCHW, DimsOrder::NHWC});
    }
};

//
// SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
//

template <typename ConcreteOp>
class SameInOutDimsOrder_CHW_HWC_NCHW_NHWC :
        public mlir::OpTrait::TraitBase<ConcreteOp, SameInOutDimsOrder_CHW_HWC_NCHW_NHWC> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameInOutSpecificDimsOrder(op, {DimsOrder::CHW, DimsOrder::HWC, DimsOrder::NCHW, DimsOrder::NHWC});
    }

    static void inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info) {
        inferLayoutInfoSameInOutSpecificDimsOrder(info,
                                                  {DimsOrder::CHW, DimsOrder::HWC, DimsOrder::NCHW, DimsOrder::NHWC});
    }
};

//
// AnyDimsOrder
//

template <typename ConcreteOp>
class AnyDimsOrder : public mlir::OpTrait::TraitBase<ConcreteOp, AnyDimsOrder> {
public:
    static void inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo&) {
    }
};

//
// isPureViewOp
//

bool isPureViewOp(mlir::Operation* op);

//
// SoftwareLayerOpInterface
//

struct KernelInfo final {
    SmallVector<mlir::Attribute> args;
    SmallString entryName;
    SmallString sourceFileName;
};

}  // namespace IERT
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/IERT/generated/ops_interfaces.hpp.inc>
