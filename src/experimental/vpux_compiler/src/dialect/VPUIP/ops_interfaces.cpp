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

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
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
        return errorAt(op, "Operation '{0}' doesn't implement VPUIP UPATask interface", op->getName());
    }

    auto upaTask = mlir::dyn_cast<UPATaskOpInterface>(op);
    if (upaTask == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement VPUIP UPATask interface", op->getName());
    }

    auto layer = mlir::dyn_cast<LayerInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
    }

    auto inputs = layer.getInputs();
    auto outputs = layer.getOutputs();

    for (auto val : concat<mlir::Value>(inputs, outputs)) {
        auto type = val.getType().cast<mlir::MemRefType>();
        auto mem = getPhysicalMemory(type);

        if (mlir::failed(mem)) {
            return errorAt(op, "Unsupported memory space '{0}'", type.getMemorySpace());
        }

        if (mem.getValue() == PhysicalMemory::CMX_NN) {
            return errorAt(op, "Can't operate with '{0}' PhysicalMemory", mem.getValue());
        }

        const auto shape = getShape(val);
        if (shape.size() != 4) {
            return errorAt(op, "Got unsupported shape '{0}', only 4D is supported", shape);
        }

        const auto order = DimsOrder::fromValue(val);
        if (!order.hasValue()) {
            return errorAt(op, "Type '{0}' has unknown DimsOrder", type);
        }
        if (order.getValue() != DimsOrder::NCHW && order.getValue() != DimsOrder::NHWC) {
            return errorAt(op, "Got unsupported input DimsOrder '{0}', only NCHW and NHWC are supported", order);
        }

        const auto elemSize = getElemTypeSize(type);
        const auto strides = getStrides(val);
        const auto memShape = order->toMemoryOrder(shape);
        const auto memStrides = order->toMemoryOrder(strides);

        const auto strideReqs = StrideReqs().add(DimStrideReq::compact(MemDim(0)));

        if (!strideReqs.checkStrides(memStrides, elemSize, memShape)) {
            return errorAt(op, "Memory strides '{0}' do not match requirements '{1}'", memStrides, strideReqs);
        }
    }

    if (upaTask.maxShaves().hasValue()) {
        auto resources = IERT::RunTimeResourcesOp::getFromModule(op->getParentOfType<mlir::ModuleOp>());
        if (resources == nullptr) {
            return errorAt(op, "Missing IERT run-time resources definition");
        }

        auto available = resources.getAvailableExecutor(
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
// getTaskEffects
//

void vpux::VPUIP::getTaskEffects(mlir::Operation* op, SmallVectorImpl<MemoryEffect>& effects) {
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
// SameShape
//

mlir::LogicalResult vpux::VPUIP::verifySameShape(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifySameShape");

    auto layer = mlir::dyn_cast<LayerInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
    }

    const auto input = layer.getInputs().front();
    const auto output = layer.getOutputs().front();

    const auto inShape = getShape(input);
    const auto outShape = getShape(output);

    if (inShape != outShape) {
        return errorAt(op, "Input shape '{0}' doesn't match with output '{1}'", inShape, outShape);
    }

    return mlir::success();
}

//
// SameElementType
//

mlir::LogicalResult vpux::VPUIP::verifySameElementType(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifySameElementType");

    auto layer = mlir::dyn_cast<LayerInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
    }

    const auto input = layer.getInputs().front();
    const auto output = layer.getOutputs().front();

    const auto inElemType = input.getType().cast<mlir::ShapedType>().getElementType();
    const auto outElemType = output.getType().cast<mlir::ShapedType>().getElementType();

    if (inElemType != outElemType) {
        return errorAt(op, "Input element type '{0}' doesn't match with output '{1}'", inElemType, outElemType);
    }

    return mlir::success();
}

//
// SameDimsOrder
//

mlir::LogicalResult vpux::VPUIP::verifySameDimsOrder(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifySameDimsOrder");

    auto layer = mlir::dyn_cast<LayerInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
    }

    const auto input = layer.getInputs().front();
    const auto output = layer.getOutputs().front();

    const auto inOrder = DimsOrder::fromValue(input);
    const auto outOrder = DimsOrder::fromValue(output);

    if (inOrder != outOrder) {
        return errorAt(op, "Input DimsOrder '{0}' doesn't match with output '{1}'", inOrder, outOrder);
    }

    return mlir::success();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUIP/generated/ops_interfaces.cpp.inc>
