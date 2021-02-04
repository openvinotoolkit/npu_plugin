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
// SingleInputAndResultLayer
//

template <typename ConcreteOp>
class SingleInputAndResultLayer : public mlir::OpTrait::TraitBase<ConcreteOp, SingleInputAndResultLayer> {
public:
    SmallVector<mlir::Value> getInputs() {
        return {mlir::cast<ConcreteOp>(this->getOperation()).input()};
    }

    SmallVector<mlir::Value> getOutputs() {
        return {mlir::cast<ConcreteOp>(this->getOperation()).output()};
    }
};

//
// Layer verifiers
//

mlir::LogicalResult verifyConstant(mlir::Operation* op);
mlir::LogicalResult verifyLayer(mlir::Operation* op);
mlir::LogicalResult verifyConvertLayer(mlir::Operation* op);
mlir::LogicalResult verifySoftMaxLayer(mlir::Operation* op);

}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/core/generated/ops_interfaces.hpp.inc>
