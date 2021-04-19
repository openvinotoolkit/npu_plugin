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

    SmallVector<mlir::Value> getInputs() {
        return getRTLayerInOperand(this->getOperation());
    }

    SmallVector<mlir::Value> getOutputs() {
        return getRTLayerOutOperand(this->getOperation());
    }
};

}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/core/generated/ops_interfaces.hpp.inc>
