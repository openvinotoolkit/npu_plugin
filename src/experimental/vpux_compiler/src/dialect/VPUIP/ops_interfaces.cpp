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

#include "vpux/compiler/dialect/VPUIP/ops_interfaces.hpp"

#include "vpux/compiler/dialect/VPUIP/effects.hpp"

#include "vpux/utils/core/format.hpp"

#include <mlir/IR/StandardTypes.h>

using namespace vpux;

//
// getTaskEffects
//

void vpux::VPUIP::getTaskEffects(
        mlir::Operation* op, SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects) {
    auto task = mlir::cast<TaskOpInterface>(op);

    for (const auto input : task.inputTensors()) {
        auto inputType = input.getType().cast<mlir::MemRefType>();
        auto resource = getMemoryResource(inputType);
        effects.emplace_back(mlir::MemoryEffects::Read::get(), input, resource.getValue());
    }

    for (const auto output : task.outputTensors()) {
        auto outputType = output.getType().cast<mlir::MemRefType>();
        auto resource = getMemoryResource(outputType);
        effects.emplace_back(mlir::MemoryEffects::Write::get(), output, resource.getValue());
    }

    for (const auto waitBarrier : task.waitBarriers()) {
        effects.emplace_back(mlir::MemoryEffects::Read::get(), waitBarrier, VPUIP::BarrierResource::get());
    }

    for (const auto updateBarrier : task.updateBarriers()) {
        effects.emplace_back(mlir::MemoryEffects::Write::get(), updateBarrier, VPUIP::BarrierResource::get());
    }
}

//
// UPATaskOpInterface
//

mlir::LogicalResult vpux::VPUIP::verifyUPATask(mlir::Operation* op) {
    auto upaTask = mlir::cast<UPATaskOpInterface>(op);

    auto task = mlir::dyn_cast<TaskOpInterface>(op);
    if (task == nullptr) {
        return printTo(op->emitError(), "UPA Task {0} doesn't have TaskOpInterface", op->getName());
    }

    if (task.getTaskType() != VPUIP::TaskType::UPA) {
        return printTo(op->emitError(), "UPA Task {0} has wrong TaskType {1}", op->getName(), task.getTaskType());
    }

    if (upaTask.isTrailingSWLayer()) {
        for (auto updateBarrier : task.updateBarriers()) {
            for (auto* depOp : updateBarrier.getUsers()) {
                auto depTask = mlir::dyn_cast<VPUIP::TaskOpInterface>(depOp);

                if (depTask == nullptr) {
                    return printTo(op->emitError(), "Trailing UPA Task has non-SW dependency : {0}", *depOp);
                }

                if (depTask.getTaskType() != VPUIP::TaskType::UPA) {
                    return printTo(op->emitError(), "Trailing UPA Task has non-SW dependency : {0}", *depOp);
                }
            }
        }
    }

    return mlir::success();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUIP/generated/ops_interfaces.cpp.inc>
