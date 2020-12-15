//
// Copyright 2020 Intel Corporation.
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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/utils/core/format.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::verifyOp(DeclareConstantTensorOp op) {
    auto memref = op.memory().getType().cast<mlir::MemRefType>();
    auto mem = getPhysicalMemory(memref);

    if (mlir::failed(mem) || mem.getValue() != VPUIP::PhysicalMemory::DDR) {
        return printTo(op.emitError(), "'{0}' has unsupported result memory space '{1}'",
                       DeclareConstantTensorOp::getOperationName(), memref.getMemorySpace());
    }

    return mlir::success();
}

mlir::OpFoldResult vpux::VPUIP::DeclareConstantTensorOp::fold(ArrayRef<mlir::Attribute>) {
    return content();
}
