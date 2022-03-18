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
