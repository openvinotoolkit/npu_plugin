//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// ViewLikeOpInterface
//

mlir::Value VPUIP::DistributedCastOp::getViewSource() {
    return input();
}

//
// fold
//

mlir::OpFoldResult VPUIP::DistributedCastOp::fold(ArrayRef<mlir::Attribute>) {
    return input().getType() == output().getType() ? input() : nullptr;
}

//
// verify
//

mlir::LogicalResult vpux::VPUIP::DistributedCastOp::verify() {
    const auto op = getOperation();
    const auto logCb = [op](const formatv_object_base& msg) {
        std::ignore = errorAt(op, "{0}", msg.str());
    };

    if (auto sparseBufferInput = input().getType().dyn_cast<VPUIP::SparseBufferType>()) {
        if (auto sparseBufferOutput = output().getType().dyn_cast<VPUIP::SparseBufferType>()) {
            const auto inputData = sparseBufferInput.getData().cast<VPUIP::DistributedBufferType>();
            const auto outputData = sparseBufferOutput.getData().cast<VPUIP::DistributedBufferType>();
            return VPU::isDistributedCastCompatible(inputData, outputData, logCb);
        } else {
            logCb(formatv("Mismatch between types for input and output. "
                          "If input is SparseBufferType then output must be of same type."));
            return mlir::failure();
        }
    }

    const auto inDistributedType = input().getType().cast<VPUIP::DistributedBufferType>();
    const auto outDistributedType = output().getType().cast<VPUIP::DistributedBufferType>();

    return VPU::isDistributedCastCompatible(inDistributedType, outDistributedType, logCb);
}
