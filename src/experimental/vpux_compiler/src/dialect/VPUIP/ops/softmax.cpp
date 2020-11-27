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

mlir::ValueRange vpux::VPUIP::SoftMaxUPAOp::getInputs() {
    return inputTensors();
}

mlir::ValueRange vpux::VPUIP::SoftMaxUPAOp::getOutputs() {
    return outputTensors();
}

mlir::LogicalResult vpux::VPUIP::verifyOp(SoftMaxUPAOp op) {
    const auto srcType = op.getSrcType().cast<mlir::MemRefType>();
    const auto dstType = op.getDstType().cast<mlir::MemRefType>();

    const auto srcMem = getPhysicalMemory(srcType);
    const auto dstMem = getPhysicalMemory(dstType);

    if (mlir::failed(srcMem)) {
        return printTo(op.emitError(), "Input tensor for Operation '{0}' has unsupported memory space '{1}'",
                       op.getOperation()->getName(), srcType.getMemorySpace());
    }
    if (mlir::failed(dstMem)) {
        return printTo(op.emitError(), "Output tensor for Operation '{0}' has unsupported memory space '{1}'",
                       op.getOperation()->getName(), dstType.getMemorySpace());
    }

    if (srcMem.getValue() != PhysicalMemory::DDR && srcMem.getValue() != PhysicalMemory::CSRAM) {
        return printTo(op.emitError(), "'{0}' can't operate with '{1}' PhysicalMemory", op.getOperation()->getName(),
                       srcMem.getValue());
    }
    if (srcMem.getValue() != PhysicalMemory::DDR && srcMem.getValue() != PhysicalMemory::CSRAM) {
        return printTo(op.emitError(), "'{0}' can't operate with '{1}' PhysicalMemory", op.getOperation()->getName(),
                       dstMem.getValue());
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::SoftMaxUPAOp::serialize(vpux::VPUIP::BlobWriter& writer) {
    const auto axisDim = getAxisDim();

    MVCNN::SoftmaxParamsBuilder builder(writer);
    builder.add_axis(checked_cast<uint32_t>(axisDim.ind()));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(getOperation(), {paramsOff.Union(), MVCNN::SoftwareLayerParams_SoftmaxParams},
                                     checked_cast<int32_t>(maxShaves()), isTrailingSWLayer());
}
