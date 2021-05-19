//
// Copyright Intel Corporation.
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
