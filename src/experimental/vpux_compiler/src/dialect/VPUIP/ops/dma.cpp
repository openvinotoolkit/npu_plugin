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

#include <mlir/IR/StandardTypes.h>

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::verifyOp(UPADMAOp op) {
    if (op.inputTensors().size() != 1) {
        return printTo(op.emitError(),
                       "'{0}' must have 1 input tensor, got {1}",
                       UPADMAOp::getOperationName(),
                       op.inputTensors().size());
    }

    if (op.outputTensors().size() != 1) {
        return printTo(op.emitError(),
                       "'{0}' must have 1 output tensor, got {1}",
                       UPADMAOp::getOperationName(),
                       op.outputTensors().size());
    }

    const auto src = op.inputTensors().front();
    const auto dst = op.outputTensors().front();

    const auto srcType = src.getType().cast<mlir::MemRefType>();
    const auto dstType = dst.getType().cast<mlir::MemRefType>();

    const auto srcMem = MemoryLocationAttr::toPhysicalMemory(srcType);
    const auto dstMem = MemoryLocationAttr::toPhysicalMemory(dstType);

    if (srcMem == PhysicalMemory::CMX_NN || dstMem == PhysicalMemory::CMX_NN) {
        return printTo(op.emitError(),
                       "'{0}' can't copy from '{1}' to '{2}'",
                       UPADMAOp::getOperationName(),
                       srcMem,
                       dstMem);
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask
        vpux::VPUIP::UPADMAOp::serialize(vpux::VPUIP::BlobWriter& writer) {
    const auto src = inputTensors().front();
    const auto dst = outputTensors().front();

    const auto srcOff = writer.getTensor(src);
    const auto dstOff = writer.getTensor(dst);

    MVCNN::UPADMATaskBuilder builder(writer);
    builder.add_src(srcOff);
    builder.add_dst(dstOff);
    return {builder.Finish().Union(), MVCNN::SpecificTask_UPADMATask};
}
