//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/utils/asm.hpp"

using namespace vpux;

mlir::LogicalResult VPU::sameOrder(mlir::Location loc, VPU::DistributedTensorType inDistributedType,
                                   VPU::DistributedTensorType outDistributedType) {
    if (inDistributedType.getOrder() != outDistributedType.getOrder()) {
        return errorAt(loc, "Mismatch between order for input ({0}) and output ({1}).", inDistributedType.getOrder(),
                       outDistributedType.getOrder());
    }
    return mlir::success();
}

mlir::LogicalResult VPU::sameOrder(mlir::Location loc, VPUIP::DistributedBufferType inDistributedType,
                                   VPUIP::DistributedBufferType outDistributedType) {
    if (inDistributedType.getLayout() != outDistributedType.getLayout()) {
        return errorAt(loc, "Mismatch between order for input ({0}) and output ({1}).", inDistributedType.getLayout(),
                       outDistributedType.getLayout());
    }
    return mlir::success();
}

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/generated/ops.cpp.inc>
