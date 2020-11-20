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

//
// DeclareTensorOp
//

mlir::LogicalResult vpux::VPUIP::verifyOp(DeclareTensorOp op) {
    const auto location = op.location();

    if (location == MemoryLocation::ProgrammableInput ||
        location == MemoryLocation::ProgrammableOutput ||
        location == MemoryLocation::GraphFile) {
        return printTo(op.emitError(),
                       "MemoryLocation '{0}' can't be used in '{1}'",
                       location,
                       DeclareTensorOp::getOperationName());
    }

    const auto memref = op.memory().getType().cast<mlir::MemRefType>();

    if (!isMemoryCompatible(location, memref)) {
        return printTo(
                op.emitError(),
                "'{0}' location '{1}' is not compatible with memory Type '{2}'",
                DeclareTensorOp::getOperationName(),
                location,
                memref);
    }

    if (const auto offsetAttr = op.offsetAttr()) {
        const auto offset = offsetAttr.getValue().getSExtValue();
        if (offset < 0) {
            return printTo(op.emitError(),
                           "Got negative offset '{0}' for '{1}'",
                           offset,
                           DeclareTensorOp::getOperationName());
        }

        // TODO: check memory limitations
    }

    return mlir::success();
}

//
// ConfigureBarrierOp
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ConfigureBarrierOp::serialize(
        vpux::VPUIP::BlobWriter& writer) {
    const auto barrier = writer.createBarrier(this->barrier());

    MVCNN::BarrierConfigurationTaskBuilder subBuilder(writer);
    subBuilder.add_target(barrier);
    const auto subTask = subBuilder.Finish();

    MVCNN::ControllerTaskBuilder builder(writer);
    builder.add_task_type(MVCNN::ControllerSubTask_BarrierConfigurationTask);
    builder.add_task(subTask.Union());

    return {builder.Finish().Union(), MVCNN::SpecificTask_ControllerTask};
}

mlir::LogicalResult vpux::VPUIP::verifyOp(ConfigureBarrierOp op) {
    if (!op.inputTensors().empty()) {
        return printTo(op.emitError(),
                       "'{0}' must not have input tensors, got {1}",
                       ConfigureBarrierOp::getOperationName(),
                       op.inputTensors().size());
    }

    if (!op.outputTensors().empty()) {
        return printTo(op.emitError(),
                       "'{0}' must not have output tensors, got {1}",
                       ConfigureBarrierOp::getOperationName(),
                       op.outputTensors().size());
    }

    return mlir::success();
}
