//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/small_string.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>

namespace vpux {
namespace VPUIPDPU {

//
// ArchKindVPUX37XX
//

mlir::LogicalResult verifyArchKindVPUX37XX(mlir::Operation* op);

template <typename ConcreteOp>
class ArchKindVPUX37XX : public mlir::OpTrait::TraitBase<ConcreteOp, ArchKindVPUX37XX> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifyArchKindVPUX37XX(op);
    }
};

}  // namespace VPUIPDPU
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/VPUIPDPU/generated/ops_interfaces.hpp.inc>
