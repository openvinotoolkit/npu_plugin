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

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>

namespace vpux {

//
// DataOrderInfo
//

class DataOrderInfo final {
public:
    DataOrderInfo(size_t numInputs, size_t numOutputs) {
        _inputOrders.resize(numInputs);
        _outputOrders.resize(numOutputs);
    }

public:
    void setInput(size_t argNum, DimsOrder order) {
        VPUX_THROW_UNLESS(argNum < _inputOrders.size(), "Argument number {0} is out of range {1}", argNum,
                          _inputOrders.size());
        _inputOrders[argNum] = order;
    }
    void setOutput(size_t argNum, DimsOrder order) {
        VPUX_THROW_UNLESS(argNum < _outputOrders.size(), "Argument number {0} is out of range {1}", argNum,
                          _outputOrders.size());
        _outputOrders[argNum] = order;
    }

    bool hasInput(size_t argNum) const {
        VPUX_THROW_UNLESS(argNum < _inputOrders.size(), "Argument number {0} is out of range {1}", argNum,
                          _inputOrders.size());
        return _inputOrders[argNum].hasValue();
    }
    bool hasOutput(size_t argNum) const {
        VPUX_THROW_UNLESS(argNum < _outputOrders.size(), "Argument number {0} is out of range {1}", argNum,
                          _outputOrders.size());
        return _outputOrders[argNum].hasValue();
    }

    DimsOrder getInput(size_t argNum) const {
        VPUX_THROW_UNLESS(hasInput(argNum), "No value for argument {0}", argNum);
        return _inputOrders[argNum].getValue();
    }
    DimsOrder getOutput(size_t argNum) const {
        VPUX_THROW_UNLESS(hasOutput(argNum), "No value for argument {0}", argNum);
        return _outputOrders[argNum].getValue();
    }

public:
    void printFormat(llvm::raw_ostream& stream) const;

private:
    SmallVector<Optional<DimsOrder>> _inputOrders;
    SmallVector<Optional<DimsOrder>> _outputOrders;
};

void fillDataInfo(DataOrderInfo& info, size_t inNum, size_t outNum, DimsOrder mainOrder);

//
// LayerInterface
//

mlir::LogicalResult verifyLayer(mlir::Operation* op);

DataOrderInfo getLayerDataOrderInfo(mlir::Operation* op);

using InferTypeComponentsCb = FuncRef<mlir::LogicalResult(mlir::MLIRContext*, Optional<mlir::Location>,
                                                          mlir::ValueRange, mlir::DictionaryAttr, mlir::RegionRange,
                                                          SmallVectorImpl<mlir::ShapedTypeComponents>&)>;

mlir::LogicalResult inferTensorTypes(InferTypeComponentsCb componentsCb, mlir::MLIRContext* ctx,
                                     Optional<mlir::Location> loc, mlir::ValueRange operands,
                                     mlir::DictionaryAttr attrs, mlir::RegionRange regions,
                                     SmallVectorImpl<mlir::Type>& inferredTypes);

bool isCompatibleShapeAndElemType(mlir::TypeRange lhs, mlir::TypeRange rhs);

//
// RTLayerInterface
//

mlir::LogicalResult verifyRTLayer(mlir::Operation* op);

mlir::OperandRange getRTLayerInputs(mlir::Operation* op);
mlir::OperandRange getRTLayerOutputs(mlir::Operation* op);

MutableArrayRef<mlir::OpOperand> getRTLayerInOpOperands(mlir::Operation* op);
MutableArrayRef<mlir::OpOperand> getRTLayerOutOpOperands(mlir::Operation* op);

DataOrderInfo getRTLayerDataOrderInfo(mlir::Operation* op);

mlir::Value getRTLayerViewSource(mlir::Operation* op, ptrdiff_t resultInd);

mlir::LogicalResult inferRTLayerReturnTypes(mlir::ValueRange operands, size_t numResults,
                                            SmallVectorImpl<mlir::Type>& inferredReturnTypes);

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
bool isSupportedLayoutSameDimsOrder(mlir::Operation* op, DataOrderInfo& info);

template <typename ConcreteOp>
class SameDimsOrder : public mlir::OpTrait::TraitBase<ConcreteOp, SameDimsOrder> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameDimsOrder(op);
    }

    static bool isSupportedLayout(mlir::Operation* op, DataOrderInfo& info) {
        return isSupportedLayoutSameDimsOrder(op, info);
    }
};

//
// SameInOutDimsOrder
//

mlir::LogicalResult verifySameInOutDimsOrder(mlir::Operation* op);
bool isSupportedLayoutSameInOutDimsOrder(mlir::Operation* op, DataOrderInfo& info);

template <typename ConcreteOp>
class SameInOutDimsOrder : public mlir::OpTrait::TraitBase<ConcreteOp, SameInOutDimsOrder> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameInOutDimsOrder(op);
    }

    static bool isSupportedLayout(mlir::Operation* op, DataOrderInfo& info) {
        return isSupportedLayoutSameInOutDimsOrder(op, info);
    }
};

//
// SameInOutSpecificDimsOrder
//

mlir::LogicalResult verifySameInOutSpecificDimsOrder(mlir::Operation* op, ArrayRef<DimsOrder> supportedLayouts);
bool isSupportedLayoutSameInOutSpecificDimsOrder(mlir::Operation* op, DataOrderInfo& info,
                                                 ArrayRef<DimsOrder> supportedLayouts);

//
// SameInOutDimsOrder_NCHW_NHWC
//

extern const std::array<DimsOrder, 2> NCHW_NHWC;

template <typename ConcreteOp>
class SameInOutDimsOrder_NCHW_NHWC : public mlir::OpTrait::TraitBase<ConcreteOp, SameInOutDimsOrder_NCHW_NHWC> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySameInOutSpecificDimsOrder(op, NCHW_NHWC);
    }

    static bool isSupportedLayout(mlir::Operation* op, DataOrderInfo& info) {
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

    static bool isSupportedLayout(mlir::Operation* op, DataOrderInfo& info) {
        return isSupportedLayoutSameInOutSpecificDimsOrder(op, info, CHW_HWC_NCHW_NHWC);
    }
};

//
// AnyDimsOrder
//

template <typename ConcreteOp>
class AnyDimsOrder : public mlir::OpTrait::TraitBase<ConcreteOp, AnyDimsOrder> {
public:
    static bool isSupportedLayout(mlir::Operation*, DataOrderInfo&) {
        return true;
    }
};

//
// Specific per-layer verifiers
//

mlir::LogicalResult verifyConvertLayer(mlir::Operation* op);
mlir::LogicalResult verifySoftMaxLayer(mlir::Operation* op);

}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/core/generated/ops_interfaces.hpp.inc>
