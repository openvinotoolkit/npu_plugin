//
// Copyright (C) 2022 Intel Corporation
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
// verifyOp
//

mlir::LogicalResult VPUIP::verifyOp(VPUIP::DistributedCastOp op) {
    const auto inDistributedType = op.input().getType().cast<VPUIP::DistributedBufferType>();
    const auto outDistributedType = op.output().getType().cast<VPUIP::DistributedBufferType>();

    return VPU::isDistributedCastCompatible(inDistributedType, outDistributedType);
}
