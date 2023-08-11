//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/cost_model_utils.hpp"

using namespace vpux;

VPUNN::VPUTensor getVPUNNTensor(ShapeRef tensorShape, VPUNN::DataType dataType) {
    return VPUNN::VPUTensor({static_cast<unsigned int>(tensorShape.totalSize()), 1, 1, 1}, dataType);
}

VPUNN::VPUTensor getVPUNNTensorMultiCluster(ArrayRef<Shape> tensorShapes, VPUNN::DataType dataType) {
    unsigned int totalShape = 0;
    for (size_t idx = 0; idx < tensorShapes.size(); idx++) {
        totalShape += static_cast<unsigned int>(tensorShapes[idx].totalSize());
    }
    return VPUNN::VPUTensor({totalShape, 1, 1, 1}, dataType);
}

VPUNN::VPUDevice getVPUDeviceType(VPU::ArchKind archKind) {
    switch (archKind) {
    case VPU::ArchKind::VPUX30XX:
    case VPU::ArchKind::VPUX311X:
        return VPUNN::VPUDevice::VPU_2_0;
    case VPU::ArchKind::VPUX37XX:
        return VPUNN::VPUDevice::VPU_2_7;
    default:
        VPUX_THROW("Unsupported VPU arch type: '{0}'", archKind);
    }
}

VPUNN::DataType getElementType(mlir::Type type) {
    if (type.isF16()) {
        return VPUNN::DataType::FLOAT16;
    } else if (type.isInteger(CHAR_BIT * sizeof(int8_t))) {
        return VPUNN::DataType::INT8;
    } else if (type.isUnsignedInteger(CHAR_BIT * sizeof(int8_t))) {
        return VPUNN::DataType::UINT8;
    } else if (auto qType = type.dyn_cast<mlir::quant::QuantizedType>()) {
        if (qType.getStorageTypeIntegralWidth() == 8) {
            return qType.isSigned() ? VPUNN::DataType::INT8 : VPUNN::DataType::UINT8;
        }
    }
    // default until support for more types introduced
    return VPUNN::DataType::BFLOAT16;
}

size_t calculateMultiClusterDMACost(mlir::Value innerOperand, VPUNN::DataType inElemType, VPUNN::DataType outElemType,
                                    VPU::ArchKind archKind, std::shared_ptr<VPUNN::VPUCostModel> costModel) {
    auto operandType = innerOperand.getType();
    auto distributedType = operandType.dyn_cast<VPUIP::DistributedBufferType>();
    VPUX_THROW_UNLESS(distributedType != nullptr, "Unsupported operand type {0}", operandType);

    // TODO: E#66557
    // Currently, if DMA source is OVERLAPPED we're moving the overlap twice. Once that is optimized,
    // we might need to update the cost here as well
    auto perClusterShapes = distributedType.getPerClusterMemoryShapes();

    return static_cast<size_t>(costModel->DMA(getVPUDeviceType(archKind),
                                              {getVPUNNTensorMultiCluster(perClusterShapes, inElemType)},
                                              {getVPUNNTensorMultiCluster(perClusterShapes, outElemType)}));
}

bool extraDMAsRequired(mlir::Value innerOperand) {
    if (auto inputType = innerOperand.getType().dyn_cast<VPUIP::DistributedBufferType>()) {
        auto distribution = inputType.getDistribution();
        auto distributionMode = distribution.mode().getValue();
        return distributionMode == VPU::DistributionMode::SEGMENTED ||
               distributionMode == VPU::DistributionMode::OVERLAPPED;
    }
    return false;
}

size_t vpux::getDMACost(mlir::Value input, mlir::Value output, VPU::ArchKind archKind,
                        std::shared_ptr<VPUNN::VPUCostModel> costModel) {
    auto inElemType = getElementType(input.getType().cast<vpux::NDTypeInterface>().getElementType());
    auto outElemType = getElementType(output.getType().cast<vpux::NDTypeInterface>().getElementType());

    if (auto nceClusterTiling = output.getDefiningOp()->getParentOfType<VPUIP::NCEClusterTilingOp>()) {
        auto parentInput = *nceClusterTiling.getInputs().begin();
        auto parentOutput = *nceClusterTiling.getOutputs().begin();

        if (extraDMAsRequired(parentInput)) {
            return calculateMultiClusterDMACost(parentInput, inElemType, outElemType, archKind, costModel);
        }

        if (extraDMAsRequired(parentOutput)) {
            return calculateMultiClusterDMACost(parentOutput, inElemType, outElemType, archKind, costModel);
        }
    }

    auto inputShape = getShape(input);
    auto outputShape = getShape(output);

    auto cost = costModel->DMA(getVPUDeviceType(archKind), {getVPUNNTensor(inputShape, inElemType)},
                               {getVPUNNTensor(outputShape, outElemType)});

    return static_cast<size_t>(cost);
}

size_t vpux::getDPUCost(mlir::Operation* op) {
    // costs for DPU calculated during workload generation, re-use

    if (op->hasAttr(DPUCost)) {
        auto cost = op->getAttr(DPUCost).cast<mlir::IntegerAttr>().getValue().getSExtValue();
        return checked_cast<size_t>(cost);
    }

    VPUX_THROW("Op {0} has no atrribute {1}", op->getLoc(), DPUCost);
}

size_t vpux::getAsyncExecuteCycleBegin(mlir::async::ExecuteOp op) {
    if (!op->hasAttr(cycleBegin)) {
        Logger::global().trace("Attribute '{0}' not present in async.execute '{1}'", cycleBegin, op);
        return 0;
    }
    return checked_cast<size_t>(op->getAttr(cycleBegin).cast<mlir::IntegerAttr>().getValue().getSExtValue());
}

size_t vpux::getAsyncExecuteCycleEnd(mlir::async::ExecuteOp op) {
    if (!op->hasAttr(cycleEnd)) {
        Logger::global().trace("Attribute '{0}' not present in async.execute '{1}'", cycleEnd, op);
        return 0;
    }
    return checked_cast<size_t>(op->getAttr(cycleEnd).cast<mlir::IntegerAttr>().getValue().getSExtValue());
}

size_t vpux::calculateCopyCycles(mlir::Operation* innerOp, VPU::ArchKind archKind,
                                 const std::shared_ptr<VPUNN::VPUCostModel> costModel) {
    if (auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(innerOp)) {
        return checked_cast<size_t>(getDMACost(copyOp.input(), copyOp.output(), archKind, costModel));
    } else if (auto copyOp = mlir::dyn_cast<VPUIP::NNDMAOp>(innerOp)) {
        return checked_cast<size_t>(getDMACost(copyOp.input(), copyOp.output(), archKind, costModel));
    } else if (auto copyOp = mlir::dyn_cast<VPUIP::DepthToSpaceDMAOp>(innerOp)) {
        return checked_cast<size_t>(getDMACost(copyOp.input(), copyOp.output(), archKind, costModel));
    } else if (auto copyOp = mlir::dyn_cast<VPUIP::SpaceToDepthDMAOp>(innerOp)) {
        return checked_cast<size_t>(getDMACost(copyOp.input(), copyOp.output(), archKind, costModel));
    } else if (auto copyOp = mlir::dyn_cast<VPUIP::PerAxisTileDMAOp>(innerOp)) {
        return checked_cast<size_t>(getDMACost(copyOp.input(), copyOp.output(), archKind, costModel));
    } else if (auto copyOp = mlir::dyn_cast<VPUIP::TimestampOp>(innerOp)) {
        return checked_cast<size_t>(getDMACost(copyOp.output(), copyOp.output(), archKind, costModel));
    } else if (auto copyOp = mlir::dyn_cast<VPUIP::PermuteDMAOp>(innerOp)) {
        return checked_cast<size_t>(getDMACost(copyOp.input(), copyOp.output(), archKind, costModel));
    } else if (auto copyOp = mlir::dyn_cast<VPUIP::ExpandDMAOp>(innerOp)) {
        return checked_cast<size_t>(getDMACost(copyOp.input(), copyOp.output(), archKind, costModel));
    } else if (auto copyOp = mlir::dyn_cast<VPUIP::UpsamplingDMAOp>(innerOp)) {
        return checked_cast<size_t>(getDMACost(copyOp.input(), copyOp.output(), archKind, costModel));
    }
    return 0;
}

vpux::Byte vpux::getSwKernelRunTotalAllocSize(VPUIP::SwKernelRun swKernelRun, ArrayRef<mlir::Value> inputs,
                                              ArrayRef<mlir::Value> outputBuffs,
                                              SmallVector<mlir::Value>& inputsForKernelRun,
                                              SmallVector<mlir::Value>& outputsForKernelRun) {
    const auto insSize = inputs.size();
    const auto outsSize = outputBuffs.size();
    const auto kernelOpArgsCount = insSize + outsSize;
    auto totalSwKernelRunSize = vpux::Byte(0);

    for (auto arg : swKernelRun.args()) {
        auto blkArg = arg.dyn_cast_or_null<mlir::BlockArgument>();
        if (blkArg == nullptr) {
            continue;
        }

        auto id = blkArg.getArgNumber();
        VPUX_THROW_UNLESS(id < kernelOpArgsCount,
                          "Index '{0}' of argument of Kernel.Run operation is out of range {1}'", id,
                          kernelOpArgsCount);
        mlir::Value buffer;
        if (id < insSize) {
            buffer = inputs[id];
            inputsForKernelRun.push_back(buffer);
        } else {
            buffer = outputBuffs[id - insSize];
            outputsForKernelRun.push_back(buffer);
        }
        totalSwKernelRunSize += buffer.getType().cast<vpux::NDTypeInterface>().getCompactAllocSize();
    }
    return totalSwKernelRunSize;
}

size_t vpux::getShaveActCycleForSwKernelOp(VPUIP::SwKernelOp swKernelOp, VPU::ArchKind arch,
                                           ArrayRef<mlir::Value> inputs, ArrayRef<mlir::Value> outputBuffs,
                                           const std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    SmallVector<mlir::Value> inputsForLargestKernelRun{inputs[0]};
    SmallVector<mlir::Value> outputsForLargestKernelRun{outputBuffs[0]};
    auto largestSwKernelRunSize = vpux::Byte(0);
    auto swKernelRuns = swKernelOp.body().getOps<VPUIP::SwKernelRun>();

    // SwKernelOp can have multiple SWKernelRun which could further be distributed on 2 ACTShaves in parallel
    // In such case use the largest SwKernelRun to calculate the cycle cost
    if (std::distance(swKernelRuns.begin(), swKernelRuns.end()) > 1) {
        for (auto&& kernelRun : swKernelRuns) {
            SmallVector<mlir::Value> inputsForKernelRun;
            SmallVector<mlir::Value> outputsForKernelRun;
            auto swKernelRunSize = getSwKernelRunTotalAllocSize(kernelRun, inputs, outputBuffs, inputsForKernelRun,
                                                                outputsForKernelRun);
            if (largestSwKernelRunSize < swKernelRunSize) {
                largestSwKernelRunSize = swKernelRunSize;
                inputsForLargestKernelRun = inputsForKernelRun;
                outputsForLargestKernelRun = outputsForKernelRun;
            }
        }
    }

    // For now, the inputs and outputs will be just 1, but in future we may have SWKernels that may have any number
    // of inputs and outputs In such cases the layer can be formed using a list/vector of inputs and outputs
    auto largestInputNdType = inputsForLargestKernelRun[0].getType().cast<vpux::NDTypeInterface>();
    auto largestOutputNdType = outputsForLargestKernelRun[0].getType().cast<vpux::NDTypeInterface>();

    auto inputTensor =
            getVPUNNTensor(largestInputNdType.getShape(), getElementType(largestInputNdType.getElementType()));
    auto outputTensor =
            getVPUNNTensor(largestOutputNdType.getShape(), getElementType(largestOutputNdType.getElementType()));

    auto strKernelOp = swKernelOp.kernelFunction().getLeafReference().str();
    std::unique_ptr<VPUNN::SWOperation> vpunnLayer;
    if (strKernelOp.find("SoftMax") != std::string::npos) {
        vpunnLayer = std::make_unique<VPUNN::SHVSoftmax>(getVPUDeviceType(arch), inputTensor, outputTensor);
    } else if (strKernelOp.find("MVN") != std::string::npos) {
        vpunnLayer = std::make_unique<VPUNN::SHVMVN>(getVPUDeviceType(arch), inputTensor, outputTensor);
    } else if (strKernelOp.find("Tanh") != std::string::npos) {
        vpunnLayer = std::make_unique<VPUNN::SHVTanh>(getVPUDeviceType(arch), inputTensor, outputTensor);
    } else {
        vpunnLayer = nullptr;
    }
    return vpunnLayer != nullptr ? costModel->SHAVE(*vpunnLayer) : 1;
}

size_t vpux::calculateShaveActCycles(VPUIP::SwKernelOp swKernelOp,
                                     const std::shared_ptr<VPUNN::VPUCostModel>& costModel, VPU::ArchKind arch) {
    auto inputNdType = swKernelOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    auto outputNdType = swKernelOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto inputElemType = inputNdType.getElementType();
    auto outputElemType = outputNdType.getElementType();

    // CostModel does not support F32/SI32 layers
    if (inputElemType.isF32() || outputElemType.isF32()) {
        return 1;
    }
    if (inputElemType.isSignedInteger(32) || outputElemType.isSignedInteger(32)) {
        return 1;
    }
    auto inputs = to_small_vector(swKernelOp.inputs());
    auto outputs = to_small_vector(swKernelOp.output_buffs());

    // In case the parent is a TilingOp, the Layer could be distributed
    // In such case updated the inputTensor and outputTensor with the biggest perCluster Shape
    if (auto parentOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp())) {
        inputs = to_small_vector(parentOp.inputs());
        outputs = to_small_vector(parentOp.output_buffs());
    }
    return getShaveActCycleForSwKernelOp(swKernelOp, arch, inputs, outputs, costModel);
}
