//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/dialect/EMU/graph-schema/blob_writer.hpp"

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
