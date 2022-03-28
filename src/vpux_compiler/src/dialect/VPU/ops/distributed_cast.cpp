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
    const auto logCb = [op](const llvm::formatv_object_base& msg) {
        std::ignore = errorAt(op, "{0}", msg.str());
    };

    const auto inDistributedType = op.input().getType().cast<VPU::DistributedTensorType>();
    const auto outDistributedType = op.output().getType().cast<VPU::DistributedTensorType>();

    return isDistributedCastCompatible(inDistributedType, outDistributedType, logCb);
}
