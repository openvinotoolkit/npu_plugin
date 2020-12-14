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

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/effects.hpp"

#include "vpux/utils/core/format.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

//
// verifyUPATask
//

mlir::LogicalResult vpux::VPUIP::verifyUPATask(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifyUPATask");

    auto task = mlir::dyn_cast<TaskOpInterface>(op);
    if (task == nullptr) {
        return printTo(op->emitError(), "Operation '{0}' doesn't implement VPUIP UPATask interface", op->getName());
    }

    auto upaTask = mlir::dyn_cast<UPATaskOpInterface>(op);
    if (upaTask == nullptr) {
        return printTo(op->emitError(), "Operation '{0}' doesn't implement VPUIP UPATask interface", op->getName());
    }

    auto layer = mlir::dyn_cast<LayerInterface>(op);
    if (layer == nullptr) {
        return printTo(op->emitError(), "Operation '{0}' doesn't implement Layer interface", op->getName());
    }

    auto inputs = layer.getInputs();
    auto outputs = layer.getOutputs();

    for (auto val : concat<mlir::Value>(inputs, outputs)) {
        auto type = val.getType().cast<mlir::MemRefType>();
        auto mem = getPhysicalMemory(type);

        if (mlir::failed(mem)) {
            return printTo(op->emitError(), "Operation '{0}' has argument with unsupported memory space '{1}'",
                           op->getName(), type.getMemorySpace());
        }

        if (mem.getValue() == PhysicalMemory::CMX_NN) {
            return printTo(op->emitError(), "'{0}' can't operate with '{1}' PhysicalMemory", op->getName(),
                           mem.getValue());
        }
    }

    if (upaTask.maxShaves().hasValue()) {
        auto resources = IERT::RunTimeResourcesOp::getFromModule(op->getParentOfType<mlir::ModuleOp>());
        if (resources == nullptr) {
            return printTo(op->emitError(), "Missing IERT run-time resources definition");
        }

        auto available = resources.getAvailableExecutor(
                VPUIP::PhysicalProcessorAttr::get(VPUIP::PhysicalProcessor::SHAVE_UPA, op->getContext()));
        if (available == nullptr) {
            return printTo(op->emitError(), "SHAVE_UPA executor is not avaialble in run-time");
        }
        if (upaTask.maxShaves().getValue() > available.count()) {
            return printTo(op->emitError(), "Operation 'maxShaves' attribute '{0}' exceeds available count '{1}'",
                           upaTask.maxShaves(), available.count());
        }
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
// getTaskEffects
//

void vpux::VPUIP::getTaskEffects(
        mlir::Operation* op, SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in getTaskEffects");

    if (auto layer = mlir::dyn_cast<LayerInterface>(op)) {
        for (const auto input : layer.getInputs()) {
            auto inputType = input.getType().cast<mlir::MemRefType>();
            auto resource = getMemoryResource(inputType);
            effects.emplace_back(mlir::MemoryEffects::Read::get(), input, resource.getValue());
        }

        for (const auto output : layer.getOutputs()) {
            auto outputType = output.getType().cast<mlir::MemRefType>();
            auto resource = getMemoryResource(outputType);
            effects.emplace_back(mlir::MemoryEffects::Write::get(), output, resource.getValue());
        }
    }

    auto task = mlir::dyn_cast<TaskOpInterface>(op);
    VPUX_THROW_UNLESS(task != nullptr, "Got non Task Operation '{0}' in getTaskEffects", op->getName());

    for (const auto waitBarrier : task.waitBarriers()) {
        effects.emplace_back(mlir::MemoryEffects::Read::get(), waitBarrier, VPUIP::BarrierResource::get());
    }

    for (const auto updateBarrier : task.updateBarriers()) {
        effects.emplace_back(mlir::MemoryEffects::Write::get(), updateBarrier, VPUIP::BarrierResource::get());
    }
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUIP/generated/ops_interfaces.cpp.inc>
