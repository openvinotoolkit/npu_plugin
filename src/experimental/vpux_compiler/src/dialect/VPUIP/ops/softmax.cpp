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

mlir::LogicalResult vpux::VPUIP::verifyOp(SoftMaxUPAOp op) {
    if (op.inputTensors().size() != 1) {
        return printTo(op.emitError(), "'{0}' must have 1 input tensor, got {1}", SoftMaxUPAOp::getOperationName(),
                       op.inputTensors().size());
    }

    if (op.outputTensors().size() != 1) {
        return printTo(op.emitError(), "'{0}' must have 1 output tensor, got {1}", SoftMaxUPAOp::getOperationName(),
                       op.outputTensors().size());
    }

    const auto src = op.inputTensors().front();
    const auto dst = op.outputTensors().front();

    const auto srcType = src.getType().cast<mlir::MemRefType>();
    const auto dstType = dst.getType().cast<mlir::MemRefType>();

    const auto srcMem = getPhysicalMemory(srcType);
    const auto dstMem = getPhysicalMemory(dstType);

    if (srcMem != PhysicalMemory::DDR && srcMem != PhysicalMemory::CSRAM) {
        return printTo(op.emitError(), "'{0}' can't operate with '{1}' PhysicalMemory",
                       SoftMaxUPAOp::getOperationName(), srcMem);
    }
    if (srcMem != PhysicalMemory::DDR && srcMem != PhysicalMemory::CSRAM) {
        return printTo(op.emitError(), "'{0}' can't operate with '{1}' PhysicalMemory",
                       SoftMaxUPAOp::getOperationName(), dstMem);
    }

    return mlir::success();
}

ShapeRef vpux::VPUIP::SoftMaxUPAOp::getInputShape() {
    const auto src = inputTensors().front();
    const auto srcType = src.getType().cast<mlir::MemRefType>();
    return ShapeRef(srcType.getShape());
}

Dim vpux::VPUIP::SoftMaxUPAOp::getAxisDim() {
    return Dim(axisInd());
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::SoftMaxUPAOp::serialize(vpux::VPUIP::BlobWriter& writer) {
    const auto axisDim = getAxisDim();

    MVCNN::SoftmaxParamsBuilder builder(writer);
    builder.add_axis(checked_cast<uint32_t>(axisDim.ind()));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(getOperation(), {paramsOff.Union(), MVCNN::SoftwareLayerParams_SoftmaxParams},
                                     checked_cast<int32_t>(maxShaves()), isTrailingSWLayer());
}
