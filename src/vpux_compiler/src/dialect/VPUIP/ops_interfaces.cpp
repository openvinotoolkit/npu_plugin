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

#include "vpux/compiler/dialect/VPUIP/ops_interfaces.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/effects.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/format.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

//
// TaskOpInterface
//

void vpux::VPUIP::getTaskEffects(mlir::Operation* op, SmallVectorImpl<MemoryEffect>& effects) {
    if (auto layer = mlir::dyn_cast<IERT::LayerOpInterface>(op)) {
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

mlir::Attribute vpux::VPUIP::getDMAEngine(uint32_t& numUnits, mlir::MLIRContext* ctx, VPUIP::DMAEngine engine) {
    numUnits = 1;
    return VPUIP::DMAEngineAttr::get(ctx, engine);
}

mlir::Attribute vpux::VPUIP::getPhysicalProcessor(uint32_t& numUnits, mlir::Operation* op,
                                                  VPUIP::PhysicalProcessor proc, Optional<int64_t> opUnits) {
    const auto procAttr = VPUIP::PhysicalProcessorAttr::get(op->getContext(), proc);

    if (opUnits.hasValue()) {
        numUnits = checked_cast<uint32_t>(opUnits.getValue());
    } else {
        auto module = op->getParentOfType<mlir::ModuleOp>();
        auto resources = IERT::RunTimeResourcesOp::getFromModule(module);
        auto available = resources.getExecutor(procAttr);
        VPUX_THROW_UNLESS(available != nullptr, "Executor for '{0}' is not available", procAttr);
        numUnits = checked_cast<uint32_t>(available.count());
    }

    return procAttr;
}

mlir::Attribute vpux::VPUIP::getTaskOpExecutor(mlir::Operation* op, uint32_t& numUnits) {
    auto task = mlir::cast<VPUIP::TaskOpInterface>(op);
    const auto taskType = task.getTaskType();

    switch (taskType) {
    case VPUIP::TaskType::UPADMA:
        return VPUIP::getDMAEngine(numUnits, op->getContext(), VPUIP::DMAEngine::DMA_UPA);
    case VPUIP::TaskType::NNDMA:
        return VPUIP::getDMAEngine(numUnits, op->getContext(), VPUIP::DMAEngine::DMA_NN);
    case VPUIP::TaskType::NCE2:
        return VPUIP::getPhysicalProcessor(numUnits, op, VPUIP::PhysicalProcessor::NCE_Cluster, 1);
    case VPUIP::TaskType::ACTShave:
        return VPUIP::getPhysicalProcessor(numUnits, op, VPUIP::PhysicalProcessor::SHAVE_NN, 1);
    case VPUIP::TaskType::UPA: {
        auto upaTask = mlir::cast<VPUIP::UPATaskOpInterface>(op);
        return VPUIP::getPhysicalProcessor(numUnits, op, VPUIP::PhysicalProcessor::SHAVE_UPA, upaTask.maxShaves());
    }
    default:
        VPUX_THROW("Unsupported task type '{0}'", taskType);
    }
}

//
// UPATaskOpInterface
//

mlir::LogicalResult vpux::VPUIP::verifyUPATask(mlir::Operation* op) {
    auto task = mlir::dyn_cast<TaskOpInterface>(op);
    if (task == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement VPUIP Task interface", op->getName());
    }

    auto upaTask = mlir::dyn_cast<UPATaskOpInterface>(op);
    if (upaTask == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement VPUIP UPATask interface", op->getName());
    }

    auto layer = mlir::dyn_cast<IERT::LayerOpInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement RT Layer interface", op->getName());
    }

    for (auto& operand : layer.getOpOperands()) {
        auto opVal = operand.get();
        auto type = opVal.getType().cast<mlir::MemRefType>();
        auto mem = getPhysicalMemory(type);

        if (type.getRank() == 0) {
            return errorAt(op, "SCALARS are not supported");
        }

        if (mlir::failed(mem)) {
            return errorAt(op, "Unsupported memory space '{0}'", type.getMemorySpace());
        }

        if (mem.getValue() == PhysicalMemory::CMX_NN) {
            return errorAt(op, "Can't operate with '{0}' PhysicalMemory", mem.getValue());
        }

        const auto strideReqs = StrideReqs::simple(type.getRank());

        if (!strideReqs.checkStrides(opVal)) {
            return errorAt(op, "Value '{0}' strides do not match requirements '{1}'", opVal, strideReqs);
        }
    }

    if (upaTask.maxShaves().hasValue()) {
        auto resources = IERT::RunTimeResourcesOp::getFromModule(op->getParentOfType<mlir::ModuleOp>());
        if (resources == nullptr) {
            return errorAt(op, "Missing IERT run-time resources definition");
        }

        auto available = resources.getExecutor(
                VPUIP::PhysicalProcessorAttr::get(op->getContext(), VPUIP::PhysicalProcessor::SHAVE_UPA));
        if (available == nullptr) {
            return errorAt(op, "SHAVE_UPA executor is not avaialble in run-time");
        }
        if (upaTask.maxShaves().getValue() > available.count()) {
            return errorAt(op, "maxShaves attribute '{0}' exceeds available count '{1}'", upaTask.maxShaves(),
                           available.count());
        }
    }

    if (upaTask.isTrailingSWLayer()) {
        for (auto updateBarrier : task.updateBarriers()) {
            for (auto* depOp : updateBarrier.getUsers()) {
                auto depTask = mlir::dyn_cast<VPUIP::TaskOpInterface>(depOp);

                if (depTask == nullptr) {
                    return errorAt(op, "Trailing UPA Task has non-SW dependency : '{0}'", depOp->getLoc());
                }

                if (depTask.getTaskType() != VPUIP::TaskType::UPA) {
                    return errorAt(op, "Trailing UPA Task has non-SW dependency : '{0}'", depOp->getLoc());
                }
            }
        }
    }

    return mlir::success();
}

//
// Legacy4D
//

mlir::LogicalResult vpux::VPUIP::verifyLegacy4D(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<IERT::LayerOpInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement RT Layer interface", op->getName());
    }

    for (const auto& val : layer.getOpOperands()) {
        const auto shape = getShape(val.get());
        const auto order = DimsOrder::fromValue(val.get());

        if (shape.size() != 3 && shape.size() != 4) {
            return errorAt(op, "Got unsupported shape '{0}', only 3D/4D are supported", shape);
        }

        if (shape.size() == 3) {
            if (order != DimsOrder::CHW && order != DimsOrder::HWC) {
                return errorAt(op, "Got unsupported input DimsOrder '{0}', only CHW and HWC are supported", order);
            }
        } else if (shape.size() == 4) {
            if (order != DimsOrder::NCHW && order != DimsOrder::NHWC) {
                return errorAt(op, "Got unsupported input DimsOrder '{0}', only NCHW and NHWC are supported", order);
            }

            if (shape.front() != 1) {
                return errorAt(op, "Batch size != 1 is not supported");
            }
        }
    }

    return mlir::success();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUIP/generated/ops_interfaces.cpp.inc>
