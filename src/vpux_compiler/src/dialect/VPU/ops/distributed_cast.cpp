//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// fold
//

mlir::OpFoldResult VPU::DistributedCastOp::fold(ArrayRef<mlir::Attribute>) {
    return input().getType() == output().getType() ? input() : nullptr;
}

//
// verifyOp
//

mlir::LogicalResult VPU::verifyOp(VPU::DistributedCastOp op) {
    const auto logCb = [op](const formatv_object_base& msg) {
        std::ignore = errorAt(op, "{0}", msg.str());
    };

    const auto inDistributedTypeInterface = op.input().getType().cast<VPU::DistributedTypeInterface>();
    const auto outDistributedTypeInterface = op.output().getType().cast<VPU::DistributedTypeInterface>();

    auto inDistributedTypes = inDistributedTypeInterface.getDistributedTypes();
    auto outDistributedTypes = outDistributedTypeInterface.getDistributedTypes();
    if (inDistributedTypes.size() == 0 || inDistributedTypes.size() != outDistributedTypes.size()) {
        return mlir::failure();
    }
    auto isCompatible = [&logCb](mlir::Type type1, mlir::Type type2) -> bool {
        return isDistributedCastCompatible(type1.cast<VPU::DistributedTensorType>(),
                                           type2.cast<VPU::DistributedTensorType>(), logCb)
                .succeeded();
    };

    return std::equal(inDistributedTypes.begin(), inDistributedTypes.end(), outDistributedTypes.begin(), isCompatible)
                   ? mlir::success()
                   : mlir::failure();
}
