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
