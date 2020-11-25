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
    const auto locale = op.locale();

    if (locale == MemoryLocation::ProgrammableInput || locale == MemoryLocation::ProgrammableOutput ||
        locale == MemoryLocation::GraphFile) {
        return printTo(op.emitError(), "MemoryLocation '{0}' can't be used in '{1}'", locale,
                       DeclareTensorOp::getOperationName());
    }

    // TODO: check localeIndex

    const auto memref = op.memory().getType().cast<mlir::MemRefType>();

    if (!isMemoryCompatible(locale, memref)) {
        return printTo(op.emitError(), "'{0}' locale '{1}' is not compatible with memory space '{2}'",
                       DeclareTensorOp::getOperationName(), locale, memref.getMemorySpace());
    }

    if (const auto dataIndexAttr = op.dataIndexAttr()) {
        const auto dataIndex = dataIndexAttr.getValue().getSExtValue();
        if (dataIndex < 0) {
            return printTo(op.emitError(), "Got negative dataIndex '{0}' for '{1}'", dataIndex,
                           DeclareTensorOp::getOperationName());
        }

        // TODO: check memory limitations
    }

    // TODO: check other offsets

    return mlir::success();
}

//
// ConfigureBarrierOp
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ConfigureBarrierOp::serialize(vpux::VPUIP::BlobWriter& writer) {
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
        return printTo(op.emitError(), "'{0}' must not have input tensors, got {1}",
                       ConfigureBarrierOp::getOperationName(), op.inputTensors().size());
    }

    if (!op.outputTensors().empty()) {
        return printTo(op.emitError(), "'{0}' must not have output tensors, got {1}",
                       ConfigureBarrierOp::getOperationName(), op.outputTensors().size());
    }

    return mlir::success();
}
