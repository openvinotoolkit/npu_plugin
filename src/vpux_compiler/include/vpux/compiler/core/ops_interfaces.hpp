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

#include "vpux/compiler/core/attributes/const_content.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>

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

    void setInput(size_t argNum, const vpux::DimsOrder& order) {
        VPUX_THROW_UNLESS(argNum < _inputOrders.size(), "Argument number {0} is out of range {1}", argNum,
                          _inputOrders.size());
        _inputOrders[argNum] = order;
    }
    void setOutput(size_t argNum, const vpux::DimsOrder& order) {
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

    const vpux::DimsOrder& getInput(size_t argNum) const {
        VPUX_THROW_UNLESS(hasInput(argNum), "No value for argument {0}", argNum);
        return _inputOrders[argNum].getValue();
    }
    const vpux::DimsOrder& getOutput(size_t argNum) const {
        VPUX_THROW_UNLESS(hasOutput(argNum), "No value for argument {0}", argNum);
        return _outputOrders[argNum].getValue();
    }

public:
    void printFormat(llvm::raw_ostream& stream) const;

private:
    SmallVector<mlir::Optional<DimsOrder>> _inputOrders;
    SmallVector<mlir::Optional<DimsOrder>> _outputOrders;
};

//
// Layer verifiers
//

mlir::LogicalResult verifyConstant(mlir::Operation* op);
mlir::LogicalResult verifyLayer(mlir::Operation* op);
mlir::LogicalResult verifyConvertLayer(mlir::Operation* op);
mlir::LogicalResult verifySoftMaxLayer(mlir::Operation* op);

//
// RTLayer
//

mlir::LogicalResult verifyRTLayerOp(mlir::Operation* op);
mlir::OperandRange getRTLayerInOperand(mlir::Operation* op);
mlir::OperandRange getRTLayerOutOperand(mlir::Operation* op);
MutableArrayRef<mlir::OpOperand> getRTLayerInOpOperands(mlir::Operation* op);
MutableArrayRef<mlir::OpOperand> getRTLayerOutOpOperands(mlir::Operation* op);
DataOrderInfo getRTLayerDataOrderInfo(mlir::Operation* op);
mlir::Value getRTLayerViewSource(mlir::Operation* op, ptrdiff_t resultInd);
mlir::LogicalResult inferRTLayerReturnTypes(mlir::ValueRange operands, size_t numResults,
                                            SmallVectorImpl<mlir::Type>& inferredReturnTypes);

template <typename ConcreteOp>
class RTLayer : public mlir::OpTrait::TraitBase<ConcreteOp, RTLayer> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifyRTLayerOp(op);
    }

    mlir::Value getViewSource(ptrdiff_t resultInd = 0) {
        return getRTLayerViewSource(this->getOperation(), resultInd);
    }

    mlir::OperandRange getInputs() {
        return getRTLayerInOperand(this->getOperation());
    }

    mlir::OperandRange getOutputs() {
        return getRTLayerOutOperand(this->getOperation());
    }

    auto getOpOperands() {
        return concat<mlir::OpOperand>(getInOpOperands(), getOutOpOperands());
    }

    MutableArrayRef<mlir::OpOperand> getInOpOperands() {
        return getRTLayerInOpOperands(this->getOperation());
    }

    MutableArrayRef<mlir::OpOperand> getOutOpOperands() {
        return getRTLayerOutOpOperands(this->getOperation());
    }

    DataOrderInfo getDataOrderInfo() {
        return getRTLayerDataOrderInfo(this->getOperation());
    }
};

}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/core/generated/ops_interfaces.hpp.inc>
