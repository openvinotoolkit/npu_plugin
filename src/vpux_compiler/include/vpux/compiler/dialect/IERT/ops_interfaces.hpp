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

#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"

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

IE::DataOrderInfo getLayerDataOrderInfo(mlir::Operation* op);

mlir::Value getLayerViewSource(mlir::Operation* op, ptrdiff_t resultInd);

mlir::LogicalResult inferLayerReturnTypes(mlir::ValueRange operands, size_t numResults,
                                          SmallVectorImpl<mlir::Type>& inferredReturnTypes);

//
// LayerInfoDialectInterface
//

class LayerInfoDialectInterface : public mlir::DialectInterface::Base<LayerInfoDialectInterface> {
public:
    explicit LayerInfoDialectInterface(mlir::Dialect* dialect): Base(dialect) {
    }

    virtual mlir::Attribute getExecutor(mlir::Operation* op, uint32_t& numUnits) const = 0;
};

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
bool isSupportedLayoutSameDimsOrder(mlir::Operation* op, IE::DataOrderInfo& info);

template <typename ConcreteOp>
class SameDimsOrder : public mlir::OpTrait::TraitBase<ConcreteOp, SameDimsOrder> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameDimsOrder(op);
    }

    static bool isSupportedLayout(mlir::Operation* op, IE::DataOrderInfo& info) {
        return isSupportedLayoutSameDimsOrder(op, info);
    }
};

//
// SameInOutDimsOrder
//

mlir::LogicalResult verifySameInOutDimsOrder(mlir::Operation* op);
bool isSupportedLayoutSameInOutDimsOrder(mlir::Operation* op, IE::DataOrderInfo& info);

template <typename ConcreteOp>
class SameInOutDimsOrder : public mlir::OpTrait::TraitBase<ConcreteOp, SameInOutDimsOrder> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameInOutDimsOrder(op);
    }

    static bool isSupportedLayout(mlir::Operation* op, IE::DataOrderInfo& info) {
        return isSupportedLayoutSameInOutDimsOrder(op, info);
    }
};

//
// SameInOutDimsOrder_NCHW_NHWC
//

extern const std::array<DimsOrder, 2> NCHW_NHWC;

mlir::LogicalResult verifySameInOutSpecificDimsOrder(mlir::Operation* op, ArrayRef<DimsOrder> supportedLayouts);
bool isSupportedLayoutSameInOutSpecificDimsOrder(mlir::Operation* op, IE::DataOrderInfo& info,
                                                 ArrayRef<DimsOrder> supportedLayouts);

template <typename ConcreteOp>
class SameInOutDimsOrder_NCHW_NHWC : public mlir::OpTrait::TraitBase<ConcreteOp, SameInOutDimsOrder_NCHW_NHWC> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameInOutSpecificDimsOrder(op, NCHW_NHWC);
    }

    static bool isSupportedLayout(mlir::Operation* op, IE::DataOrderInfo& info) {
        return isSupportedLayoutSameInOutSpecificDimsOrder(op, info, NCHW_NHWC);
    }
};

//
// SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
//

extern const std::array<DimsOrder, 4> CHW_HWC_NCHW_NHWC;

template <typename ConcreteOp>
class SameInOutDimsOrder_CHW_HWC_NCHW_NHWC :
        public mlir::OpTrait::TraitBase<ConcreteOp, SameInOutDimsOrder_CHW_HWC_NCHW_NHWC> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameInOutSpecificDimsOrder(op, CHW_HWC_NCHW_NHWC);
    }

    static bool isSupportedLayout(mlir::Operation* op, IE::DataOrderInfo& info) {
        return isSupportedLayoutSameInOutSpecificDimsOrder(op, info, CHW_HWC_NCHW_NHWC);
    }
};

//
// AnyDimsOrder
//

template <typename ConcreteOp>
class AnyDimsOrder : public mlir::OpTrait::TraitBase<ConcreteOp, AnyDimsOrder> {
public:
    static bool isSupportedLayout(mlir::Operation*, IE::DataOrderInfo&) {
        return true;
    }
};

}  // namespace IERT
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/IERT/generated/ops_interfaces.hpp.inc>
