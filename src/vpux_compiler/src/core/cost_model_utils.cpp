//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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

    auto perClusterShapes = distributedType.getPerClusterComputeShapes();

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
