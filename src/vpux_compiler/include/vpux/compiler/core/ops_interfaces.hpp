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

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>

namespace vpux {

//
// DeclarationOp
//

template <typename ConcreteOp>
class DeclarationOp : public mlir::OpTrait::TraitBase<ConcreteOp, DeclarationOp> {
    static mlir::LogicalResult verifyTrait(mlir::Operation*) {
        static_assert(ConcreteOp::template hasTrait<mlir::OpTrait::ZeroOperands>(),
                      "Expected operation to take zero operands");
        static_assert(ConcreteOp::template hasTrait<mlir::OpTrait::OneResult>(),
                      "Expected operation to produce one result");

        return mlir::success();
    }
};

//
// DotInterface
//

enum class DotNodeColor { NONE, RED, GREEN, ORANGE, BLUE, AQUA, AQUAMARINE };

}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/core/generated/ops_interfaces.hpp.inc>
