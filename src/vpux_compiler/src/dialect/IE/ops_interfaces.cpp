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

#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

mlir::LogicalResult IE::verifyIELayerOp(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifyRTLayerOp");

    auto isTensorized = std::all_of(op->getOperands().begin(), op->getOperands().end(), [](mlir::Value type) {
        return type.getType().isa<mlir::RankedTensorType>();
    });

    isTensorized &= std::all_of(op->getResults().begin(), op->getResults().end(), [](mlir::Value type) {
        return type.getType().isa<mlir::RankedTensorType>();
    });

    if (!isTensorized) {
        return errorAt(op, "Operation '{0}' is not a IE Layer, it operates with non Tensor types", op->getName());
    }

    return mlir::success();
}
