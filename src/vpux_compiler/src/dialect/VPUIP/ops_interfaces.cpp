//
// Copyright 2020 Intel Corporation.
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
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/effects.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/utils/extentions.hpp"

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
        return errorAt(op, "Operation '{0}' doesn't implement VPUIP Task interface", op->getName());
    }

    auto upaTask = mlir::dyn_cast<UPATaskOpInterface>(op);
    if (upaTask == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement VPUIP UPATask interface", op->getName());
    }

    auto layer = mlir::dyn_cast<LayerInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
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

    auto inputs = layer.getInputs();

    const auto firstInput = inputs.front();
    const auto mainShape = getShape(firstInput);

    for (const auto& val : layer.getOpOperands()) {
        const auto shape = getShape(val.get());

        if (shape != mainShape) {
            return errorAt(op, "Operation's input/output shapes mismatch");
        }
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

    auto inputs = layer.getInputs();

    const auto firstInput = inputs.front();
    const auto mainElemType = firstInput.getType().cast<mlir::ShapedType>().getElementType();

    for (const auto& val : layer.getOpOperands()) {
        const auto elemType = val.get().getType().cast<mlir::ShapedType>().getElementType();

        if (elemType != mainElemType) {
            return errorAt(op, "Operation's input/output element types mismatch");
        }
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

    auto inputs = layer.getInputs();

    const auto firstInput = inputs.front();
    const auto mainOrder = DimsOrder::fromValue(firstInput);

    for (const auto& val : layer.getOpOperands()) {
        const auto order = DimsOrder::fromValue(val.get());

        if (order != mainOrder) {
            return errorAt(op, "Operation's input/output layout mismatch");
        }
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::isSupportedLayoutSameDimsOrder(mlir::Operation* op, DataOrderInfo& info) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in isSupportedLayoutSameDimsOrder");

    auto layer = mlir::dyn_cast<LayerInterface>(op);
    VPUX_THROW_UNLESS(layer != nullptr, "Operation '{0}' doesn't implement Layer interface", op->getName());

    const auto inputs = layer.getInputs();
    const auto outputs = layer.getOutputs();

    const auto inNum = inputs.size();
    const auto outNum = outputs.size();

    const auto mainOrder = info.hasInput(0) ? info.getInput(0) : DimsOrder::fromValue(inputs[0]);

    for (size_t i = 0; i < inNum; ++i) {
        if (!info.hasInput(i) || info.getInput(i) != mainOrder) {
            fillDataInfo(info, inNum, outNum, mainOrder);
            return mlir::failure();
        }
    }

    for (size_t i = 0; i < outNum; ++i) {
        if (!info.hasOutput(i) || info.getOutput(i) != mainOrder) {
            fillDataInfo(info, inNum, outNum, mainOrder);
            return mlir::failure();
        }
    }

    return mlir::success();
}

//
// SameInOutDimsOrder
//

mlir::LogicalResult vpux::VPUIP::verifySameInOutDimsOrder(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<LayerInterface>(op);
    VPUX_THROW_UNLESS(layer != nullptr, "Operation {0} is not layer", op->getName());

    const auto input = layer.getInputs()[0];
    const auto output = layer.getOutputs()[0];

    const auto inOrder = DimsOrder::fromValue(input);
    const auto outOrder = DimsOrder::fromValue(output);

    if (inOrder != outOrder) {
        return errorAt(op->getLoc(), "Operation must have the same input and output order. inL={0}, outL={1}", inOrder,
                       outOrder);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::isSupportedLayoutSameInOutDimsOrder(mlir::Operation* op, DataOrderInfo& info) {
    auto layer = mlir::dyn_cast<LayerInterface>(op);
    VPUX_THROW_UNLESS(layer != nullptr, "Operation {0} is not layer", op->getName());

    if (!info.hasInput(0) || !info.hasOutput(0)) {
        const auto intType = layer.getInputs()[0].getType();
        const auto supportedOrder = info.hasInput(0)
                                            ? info.getInput(0)
                                            : DimsOrder::fromNumDims(intType.cast<mlir::ShapedType>().getRank());

        fillDataInfo(info, 1, 1, supportedOrder);
        return mlir::failure();
    }

    if (info.getInput(0) != info.getOutput(0)) {
        info.setOutput(0, info.getInput(0));
        return mlir::failure();
    }

    return mlir::success();
}

//
// SameInOutSpecificDimsOrder
//

const std::array<DimsOrder, 2> VPUIP::NCHW_NHWC = {DimsOrder::NCHW, DimsOrder::NHWC};
const std::array<DimsOrder, 4> VPUIP::CHW_HWC_NCHW_NHWC = {DimsOrder::CHW, DimsOrder::HWC, DimsOrder::NCHW,
                                                           DimsOrder::NHWC};

mlir::LogicalResult vpux::VPUIP::verifySameInOutSpecificDimsOrder(mlir::Operation* op,
                                                                  ArrayRef<DimsOrder> supportedLayouts) {
    if (verifySameInOutDimsOrder(op).failed()) {
        return mlir::failure();
    }

    auto layerOp = mlir::dyn_cast<LayerInterface>(op);

    const auto input = layerOp.getInputs()[0];
    const auto inOrder = DimsOrder::fromValue(input);

    const auto isSupported = std::count(supportedLayouts.begin(), supportedLayouts.end(), inOrder);
    if (!isSupported) {
        return errorAt(op->getLoc(), "Operation does not support {0} layout", inOrder);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::isSupportedLayoutSameInOutSpecificDimsOrder(mlir::Operation* op, DataOrderInfo& info,
                                                                             ArrayRef<DimsOrder> supportedLayouts) {
    auto layer = mlir::dyn_cast<LayerInterface>(op);
    VPUX_THROW_UNLESS(layer != nullptr, "Operation '{0}' doesn't implement Layer interface", op->getName());

    const auto intType = layer.getInputs()[0].getType().cast<mlir::ShapedType>();
    const auto defaultOrderIt =
            std::find_if(supportedLayouts.begin(), supportedLayouts.end(), [intType](DimsOrder order) {
                return static_cast<int64_t>(order.numDims()) == intType.getRank();
            });

    VPUX_THROW_UNLESS(defaultOrderIt != supportedLayouts.end(),
                      "Layouts supported ({0}) by the operation '{1}' do not match the rank {2} of the input shape ",
                      supportedLayouts, op->getName(), intType.getRank());

    const auto defaultOrder = *defaultOrderIt;
    if (!info.hasInput(0)) {
        fillDataInfo(info, 1, 1, defaultOrder);

        return mlir::failure();
    }

    const auto mainOrder = info.getInput(0);
    const auto isSupportedLayout = std::count(supportedLayouts.begin(), supportedLayouts.end(), mainOrder);
    if (isSupportedLayout) {
        return isSupportedLayoutSameInOutDimsOrder(op, info);
    }

    fillDataInfo(info, 1, 1, defaultOrder);
    return mlir::failure();
}

//
// Legacy4D
//

mlir::LogicalResult vpux::VPUIP::verifyLegacy4D(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifySameDimsOrder");

    auto layer = mlir::dyn_cast<LayerInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
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
