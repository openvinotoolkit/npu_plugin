//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/cost_model.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/sw_utils.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"

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

VPUNN::MemoryLocation getMemoryLocation(mlir::Type type) {
    auto memKind = type.cast<vpux::NDTypeInterface>().getMemoryKind();
    if (memKind == VPU::MemoryKind::CMX_NN) {
        return VPUNN::MemoryLocation::CMX;
    }

    return VPUNN::MemoryLocation::DRAM;
}

VPUNN::Swizzling getVPUNNSwizzlingKey(mlir::Type type) {
    SmallVector<VPUNN::Swizzling> swizzlingKeyVPUNN = {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_1,
                                                       VPUNN::Swizzling::KEY_2, VPUNN::Swizzling::KEY_3,
                                                       VPUNN::Swizzling::KEY_4, VPUNN::Swizzling::KEY_5};

    auto swizzlingKey = vpux::getSwizzlingKey(type);
    VPUX_THROW_UNLESS(checked_cast<size_t>(swizzlingKey) < swizzlingKeyVPUNN.size(), "Unsupported swizzling key: '{0}'",
                      swizzlingKey);

    return swizzlingKeyVPUNN[swizzlingKey];
}

VPUNN::ActivationFunction getVPUNNActivationFunction(VPUIP::PPETaskOp ppeOp) {
    auto ppeType = ppeOp.ppe_layer_type();

    switch (ppeType) {
    case VPU::PPEMode::LRELU:
        return VPUNN::ActivationFunction::LRELU;
    case VPU::PPEMode::ADD:
        return VPUNN::ActivationFunction::ADD;
    case VPU::PPEMode::SUB:
        return VPUNN::ActivationFunction::SUB;
    case VPU::PPEMode::MULT:
        return VPUNN::ActivationFunction::MULT;
    default:
        if (ppeOp.clamp_low().has_value()) {
            auto clampLow = checked_cast<int32_t>(ppeOp.clamp_low().value());
            if (clampLow == 0) {
                return VPUNN::ActivationFunction::RELU;
            }
        }
        return VPUNN::ActivationFunction::NONE;
    }
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
        auto distributionMode = distribution.getMode().getValue();
        return distributionMode == VPU::DistributionMode::SEGMENTED ||
               distributionMode == VPU::DistributionMode::OVERLAPPED;
    }
    return false;
}

size_t vpux::getDMACost(mlir::Value input, mlir::Value output, VPU::ArchKind archKind,
                        std::shared_ptr<VPUNN::VPUCostModel> costModel) {
    auto inputType = input.getType();
    auto outputType = output.getType();

    auto inElemType = getElementType(inputType.cast<vpux::NDTypeInterface>().getElementType());
    auto outElemType = getElementType(outputType.cast<vpux::NDTypeInterface>().getElementType());

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
                               {getVPUNNTensor(outputShape, outElemType)}, getMemoryLocation(inputType),
                               getMemoryLocation(outputType));

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
    } else if (auto convertDMAOp = mlir::dyn_cast<VPUIP::ConvertDMAOp>(innerOp)) {
        return checked_cast<size_t>(getDMACost(convertDMAOp.input(), convertDMAOp.output(), archKind, costModel));
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

std::string getSwKernelOperationName(VPUIP::SwKernelOp swKernelOp) {
    auto strKernelOp = swKernelOp.kernelFunction().getLeafReference().str();

    auto prefIndex = strKernelOp.find(vpux::VPUIP::SW_KERNEL_NAME_PREFIX.str());
    VPUX_THROW_WHEN(prefIndex == std::string::npos, "Not a valid swKernelOp name - {0}", strKernelOp);
    auto prefEndIndex = prefIndex + vpux::VPUIP::SW_KERNEL_NAME_PREFIX.size();
    VPUX_THROW_WHEN(prefEndIndex > strKernelOp.size(), "Not a valid swKernelOp name length - {0}", strKernelOp);

    auto nameSize = std::string::npos;
    auto nameEndIndex = strKernelOp.find("_", prefEndIndex);
    if (nameEndIndex != std::string::npos) {
        nameSize = nameEndIndex - prefIndex;
    }

    return strKernelOp.substr(prefEndIndex, nameSize);
}

#define SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT(_NAME_STR_, _VPUNN_TYPE_)                                         \
    {                                                                                                          \
        _NAME_STR_, [](VPUNN::VPUDevice vpuDev, VPUNN::VPUTensor inputTensor, VPUNN::VPUTensor outputTensor) { \
            return std::make_unique<_VPUNN_TYPE_>(vpuDev, inputTensor, outputTensor);                          \
        }                                                                                                      \
    }

#define SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT(_NAME_STR_, _VPUNN_TYPE_)                           \
    {                                                                                              \
        _NAME_STR_, [](VPUNN::VPUDevice vpuDev, const std::vector<VPUNN::VPUTensor>& inputTensors, \
                       VPUNN::VPUTensor outputTensor) {                                            \
            return std::make_unique<_VPUNN_TYPE_>(vpuDev, inputTensors, outputTensor);             \
        }                                                                                          \
    }

std::map<std::string,
         std::function<std::unique_ptr<VPUNN::SWOperation>(VPUNN::VPUDevice, VPUNN::VPUTensor, VPUNN::VPUTensor)>>
        swKernelNameToVpunn1InputFuncMap = {
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Sigmoid", VPUNN::SHVSigmoid),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Elu", VPUNN::SHVELU),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("HardSigmoid", VPUNN::SHVHardSigmoid),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("SoftMax", VPUNN::SHVSoftmax),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Clamp", VPUNN::SHVClamp),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("FakeQuantize", VPUNN::SHVFakeQuantize),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Quantize", VPUNN::SHVQuantizeCast),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Tanh", VPUNN::SHVTanh),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Sin", VPUNN::SHVSin),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Cos", VPUNN::SHVCos),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Sqrt", VPUNN::SHVSqrt),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Sinh", VPUNN::SHVSinh),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Cosh", VPUNN::SHVCosh),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Asinh", VPUNN::SHVAsinh),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Acosh", VPUNN::SHVAcosh),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Abs", VPUNN::SHVAbs),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Atan", VPUNN::SHVAtan),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Asin", VPUNN::SHVAsin),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Acos", VPUNN::SHVAcos),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Atanh", VPUNN::SHVAtanh),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Log", VPUNN::SHVLog),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Selu", VPUNN::SHVSelu),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Gelu", VPUNN::SHVGelu),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Exp", VPUNN::SHVExp),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Floor", VPUNN::SHVFloor),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Round", VPUNN::SHVRound),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Mish", VPUNN::SHVMish),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Erf", VPUNN::SHVErf),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Negative", VPUNN::SHVNegative),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Sign", VPUNN::SHVSign),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("YuvToRgb", VPUNN::SHVYuvToRgb),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("SoftPlus", VPUNN::SHVSoftPlus),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Swish", VPUNN::SHVSwish),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("MVN", VPUNN::SHVMVN),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Ceiling", VPUNN::SHVCeiling),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Roll", VPUNN::SHVRoll),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("Gather", VPUNN::SHVGather),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("ScatterNDUpdate", VPUNN::SHVScatterNDUpdate),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("ScatterUpdate", VPUNN::SHVScatterUpdate),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("PermuteQuantize", VPUNN::SHVPermuteQuantize),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("DepthToSpace", VPUNN::SHVDepthToSpace),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("SpaceToDepth", VPUNN::SHVSpaceToDepthOp),
                SW_KERNEL_NAME_TO_VPUNN_1_IN_ELEMENT("MemPermute", VPUNN::SHVMemPermute)};

std::map<std::string, std::function<std::unique_ptr<VPUNN::SWOperation>(
                              VPUNN::VPUDevice, const std::vector<VPUNN::VPUTensor>&, VPUNN::VPUTensor)>>
        swKernelNameToVpunnVecInputFuncMap = {
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("Power", VPUNN::SHVPower),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("Add", VPUNN::SHVAdd),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("Divide", VPUNN::SHVDivide),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("SquaredDifference", VPUNN::SHVSquaredDiff),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("FloorMod", VPUNN::SHVFloorMod),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("Less", VPUNN::SHVLess),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("LessEqual", VPUNN::SHVLessEqual),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("Greater", VPUNN::SHVGreater),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("GreaterEqual", VPUNN::SHVGreaterEqual),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("LogicalOr", VPUNN::SHVLogicalOr),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("LogicalNot", VPUNN::SHVLogicalNot),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("LogicalXor", VPUNN::SHVLogicalXor),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("Multiply", VPUNN::SHVMultiply),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("And", VPUNN::SHVAnd),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("Minimum", VPUNN::SHVMinimum),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("Maximum", VPUNN::SHVMaximum),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("Subtract", VPUNN::SHVSubtract),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("NotEqual", VPUNN::SHVNotEqual),
                SW_KERNEL_NAME_TO_VPUNN_VEC_IN_ELEMENT("Equal", VPUNN::SHVEqual)};

size_t getShaveActCycleForSwKernelFunc(const std::string& swKernelName, VPU::ArchKind arch,
                                       ArrayRef<mlir::Value> inputs, ArrayRef<mlir::Value> outputs,
                                       const std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    VPUX_THROW_WHEN(inputs.empty(), "No inputs identified for op {0}", swKernelName);
    VPUX_THROW_WHEN(outputs.empty(), "No outputs identified for op {0}", swKernelName);

    auto input0NdType = inputs[0].getType().cast<vpux::NDTypeInterface>();
    auto output0NdType = outputs[0].getType().cast<vpux::NDTypeInterface>();

    auto inputTensor = getVPUNNTensor(input0NdType.getShape(), getElementType(input0NdType.getElementType()));
    auto outputTensor = getVPUNNTensor(output0NdType.getShape(), getElementType(output0NdType.getElementType()));

    auto vpuDev = getVPUDeviceType(arch);

    std::unique_ptr<VPUNN::SWOperation> vpunnLayer;

    if (swKernelNameToVpunn1InputFuncMap.find(swKernelName) != swKernelNameToVpunn1InputFuncMap.end()) {
        vpunnLayer = swKernelNameToVpunn1InputFuncMap[swKernelName](vpuDev, inputTensor, outputTensor);
    } else if (swKernelNameToVpunnVecInputFuncMap.find(swKernelName) != swKernelNameToVpunnVecInputFuncMap.end()) {
        std::vector<VPUNN::VPUTensor> inputTensors;
        for (auto& input : inputs) {
            auto inputNd = input.getType().cast<vpux::NDTypeInterface>();
            inputTensors.push_back(getVPUNNTensor(inputNd.getShape(), getElementType(inputNd.getElementType())));
        }

        vpunnLayer = swKernelNameToVpunnVecInputFuncMap[swKernelName](vpuDev, inputTensors, outputTensor);
    }

    return vpunnLayer != nullptr ? costModel->SHAVE(*vpunnLayer) : 1;
}

std::unique_ptr<VPUNN::SWOperation> vpux::getVPUNNSWKernelOp(VPU::SWOpInterface operation) {
    auto vpuDev = VPU::getVPUDeviceType(VPU::getArch(operation));

    auto output0NdType = operation->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto outputTensor = getVPUNNTensor(output0NdType.getShape(), getElementType(output0NdType.getElementType()));

    const auto operName = operation->getName().stripDialect().str();

    if (swKernelNameToVpunn1InputFuncMap.find(operName) != swKernelNameToVpunn1InputFuncMap.end()) {
        auto input0NdType = operation->getOperand(0).getType().cast<vpux::NDTypeInterface>();
        auto inputTensor = getVPUNNTensor(input0NdType.getShape(), getElementType(input0NdType.getElementType()));

        return swKernelNameToVpunn1InputFuncMap[operName](vpuDev, inputTensor, outputTensor);
    } else if (swKernelNameToVpunnVecInputFuncMap.find(operName) != swKernelNameToVpunnVecInputFuncMap.end()) {
        std::vector<VPUNN::VPUTensor> inputTensors;
        for (auto input : operation->getOperands()) {
            auto inputNd = input.getType().cast<vpux::NDTypeInterface>();
            inputTensors.push_back(getVPUNNTensor(inputNd.getShape(), getElementType(inputNd.getElementType())));
        }

        return swKernelNameToVpunnVecInputFuncMap[operName](vpuDev, inputTensors, outputTensor);
    }

    return nullptr;
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
    if (swKernelOp.inputs().empty() || swKernelOp.output_buffs().empty()) {
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

    SmallVector<mlir::Value> inputsForLargestKernelRun(inputs.begin(), inputs.end());
    SmallVector<mlir::Value> outputsForLargestKernelRun{outputs[0]};
    auto largestSwKernelRunSize = vpux::Byte(0);
    auto swKernelRuns = swKernelOp.body().getOps<VPUIP::SwKernelRun>();

    // SwKernelOp can have multiple SWKernelRun which could further be distributed on 2 ACTShaves in parallel
    // In such case use the largest SwKernelRun to calculate the cycle cost
    if (std::distance(swKernelRuns.begin(), swKernelRuns.end()) > 1) {
        for (auto&& kernelRun : swKernelRuns) {
            SmallVector<mlir::Value> inputsForKernelRun;
            SmallVector<mlir::Value> outputsForKernelRun;
            auto swKernelRunSize =
                    getSwKernelRunTotalAllocSize(kernelRun, inputs, outputs, inputsForKernelRun, outputsForKernelRun);
            if (largestSwKernelRunSize < swKernelRunSize) {
                largestSwKernelRunSize = swKernelRunSize;
                inputsForLargestKernelRun = std::move(inputsForKernelRun);
                outputsForLargestKernelRun = std::move(outputsForKernelRun);
            }
        }
    }

    auto swKernelName = getSwKernelOperationName(swKernelOp);

    return getShaveActCycleForSwKernelFunc(swKernelName, arch, inputsForLargestKernelRun, outputsForLargestKernelRun,
                                           costModel);
}

std::vector<size_t> calculateNceVariantCycles(VPUIP::NCEClusterTaskOp nceOp,
                                              const std::shared_ptr<VPUNN::VPUCostModel>& costModel, VPU::ArchKind arch,
                                              vpux::Logger log) {
    auto inputNdType = nceOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    auto outputNdType = nceOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto inputElemType = inputNdType.getElementType();
    auto outputElemType = outputNdType.getElementType();

    // CostModel does not support F32/SI32 layers
    if (inputElemType.isF32() || outputElemType.isF32()) {
        return {1};
    }
    if (inputElemType.isSignedInteger(32) || outputElemType.isSignedInteger(32)) {
        return {1};
    }

    VPUX_THROW_WHEN(mlir::isa<VPUIP::NCEClusterTilingOp>(nceOp->getParentOp()),
                    "Only single cluster task is supported by this method, op - {0}", nceOp->getLoc());

    auto input1Swizzling = getVPUNNSwizzlingKey(inputNdType);
    auto input2Swizzling = getVPUNNSwizzlingKey(nceOp->getOperand(1).getType());
    auto outputSwizzling = getVPUNNSwizzlingKey(outputNdType);

    VPUNN::ActivationFunction activationFunction = VPUNN::ActivationFunction::NONE;
    auto ppeOps = to_small_vector(nceOp.ppe().getOps<VPUIP::PPETaskOp>());
    if (!ppeOps.empty()) {
        activationFunction = getVPUNNActivationFunction(ppeOps[0]);
    }

    unsigned int outputWriteTiles = 1;
    VPUNN::ISIStrategy isiStrategy = VPUNN::ISIStrategy::CLUSTERING;

    // Check if output is distributed (multicasted) to multiple tiles (e.g. SOK)
    if (auto distributedOutput = outputNdType.dyn_cast<VPUIP::DistributedBufferType>()) {
        const auto distributionAttr = distributedOutput.getDistribution();
        const auto mode = distributionAttr.getMode().getValue();
        if (mode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED) ||
            mode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED)) {
            outputWriteTiles = distributionAttr.getNumClusters().getInt();
            isiStrategy = VPUNN::ISIStrategy::SPLIT_OVER_K;
        }
    }

    // Check if input is distributed - segmented between multiple tiles (e.g. SOH)
    if (auto distributedInput = nceOp.parent_input().getType().dyn_cast<VPUIP::DistributedBufferType>()) {
        const auto distributionAttr = distributedInput.getDistribution();
        const auto mode = distributionAttr.getMode().getValue();
        if (mode == VPU::DistributionMode::SEGMENTED) {
            isiStrategy = VPUNN::ISIStrategy::SPLIT_OVER_H;
        }
    }

    bool isWeightsSparsityEnabled = false;
    float weightsSparsityRatio = 0;
    if (auto weightsSparsityMap = nceOp.weights_sparsity_map()) {
        isWeightsSparsityEnabled = true;

        auto weightsType = nceOp.weights().getType().cast<vpux::NDTypeInterface>();
        auto weightsElemType = weightsType.getElementType();

        const auto compressionSchemeAttr = VPUIP::getCompressionSchemeAttr(weightsType);
        VPUX_THROW_WHEN(compressionSchemeAttr == nullptr, "compression_schemeAttr shouldn't be a nullptr");

        auto compressedSize = compressionSchemeAttr.getAllocSize(weightsElemType).count();
        weightsSparsityRatio = vpux::getWeightsSparsityRatio(weightsType, compressedSize);
    }

    auto isInputSparsityEnabled =
            (nceOp.input_sparsity_map() != nullptr || nceOp.input_storage_element_table() != nullptr);
    auto isOutputSparsityEnabled = (nceOp.output_sparsity_map() != nullptr);

    auto nceTaskType = nceOp.task_type();
    auto opType = getOperationType(nceTaskType);

    int64_t KX = 1, KY = 1;
    int64_t SX = 1, SY = 1;

    if (auto kernelSizeAttr = nceOp.kernel_sizeAttr()) {
        const auto kernelSize = parseIntArrayAttr<int64_t>(kernelSizeAttr);
        KX = kernelSize[Dims4D::Kernel::X.ind()];
        KY = kernelSize[Dims4D::Kernel::Y.ind()];
    }

    if (auto kernelStridesAttr = nceOp.kernel_stridesAttr()) {
        const auto kernelStrides = parseIntArrayAttr<int64_t>(kernelStridesAttr);
        SX = kernelStrides[Dims4D::Kernel::X.ind()];
        SY = kernelStrides[Dims4D::Kernel::Y.ind()];
    }

    mlir::DenseSet<int64_t> clusters;
    std::vector<size_t> nceVariantCycles;

    for (auto dpuTaskOp : nceOp.variants().getOps<VPUIP::DPUTaskOp>()) {
        clusters.insert(dpuTaskOp.cluster_id().value_or(0));

        auto mpeMode = dpuTaskOp.mpe_mode();

        const auto paddingAttr = dpuTaskOp.pad();

        const auto left = paddingAttr.getLeft().getValue().getSExtValue();
        const auto right = paddingAttr.getRight().getValue().getSExtValue();
        const auto top = paddingAttr.getTop().getValue().getSExtValue();
        const auto bottom = paddingAttr.getBottom().getValue().getSExtValue();

        const auto outStart = parseIntArrayAttr<int64_t>(dpuTaskOp.outStart());
        const auto outEnd = parseIntArrayAttr<int64_t>(dpuTaskOp.outEnd());

        VPUX_THROW_WHEN(outStart.size() != 3 || outEnd.size() != 3, "Unexpected size of outStart/End attributes");

        // DPUTask workload description is expected to have 3 elements: [W, H, C]
        const int64_t OC = outEnd[2] - outStart[2] + 1;
        const int64_t OH = outEnd[1] - outStart[1] + 1;
        const int64_t OW = outEnd[0] - outStart[0] + 1;

        auto IW = (OW - 1) * SX + KX - left - right;
        auto IH = (OH - 1) * SY + KY - top - bottom;
        auto IC = nceTaskType == VPUIP::NCETaskType::CONV || nceTaskType == VPUIP::NCETaskType::CMCONV ||
                                  nceTaskType == VPUIP::NCETaskType::FCL
                          ? inputNdType.getShape()[Dims4D::Act::C]
                          : OC;

        if (dpuTaskOp.inStart().has_value() && dpuTaskOp.inEnd().has_value()) {
            const auto inStart = parseIntArrayAttr<int64_t>(dpuTaskOp.inStart().value());
            const auto inEnd = parseIntArrayAttr<int64_t>(dpuTaskOp.inEnd().value());

            IC = inEnd[2] - inStart[2] + 1;
            IH = inEnd[1] - inStart[1] + 1;
            IW = inEnd[0] - inStart[0] + 1;
        }

        const auto inputTensor = VPUNN::VPUTensor(
                {static_cast<unsigned int>(IW), static_cast<unsigned int>(IH), static_cast<unsigned int>(IC), 1},
                getElementType(inputElemType), VPUNN::Layout::ZXY, isInputSparsityEnabled);
        const auto outputTensor = VPUNN::VPUTensor(
                {static_cast<unsigned int>(OW), static_cast<unsigned int>(OH), static_cast<unsigned int>(OC), 1},
                getElementType(outputElemType), VPUNN::Layout::ZXY, isOutputSparsityEnabled);

        VPUNN::DPUWorkload vpunnDPUWorkload;
        vpunnDPUWorkload.device = getVPUDeviceType(arch);
        vpunnDPUWorkload.op = opType;
        vpunnDPUWorkload.inputs = {inputTensor};
        vpunnDPUWorkload.outputs = {outputTensor};
        vpunnDPUWorkload.kernels = {static_cast<unsigned int>(KX), static_cast<unsigned int>(KY)};
        vpunnDPUWorkload.strides = {static_cast<unsigned int>(SX), static_cast<unsigned int>(SY)};
        vpunnDPUWorkload.padding = {static_cast<unsigned int>(top), static_cast<unsigned int>(bottom),
                                    static_cast<unsigned int>(left), static_cast<unsigned int>(right)};
        vpunnDPUWorkload.execution_order = VPU::getExecutionMode(mpeMode);
        vpunnDPUWorkload.activation_function = activationFunction;
        vpunnDPUWorkload.input_swizzling = {input1Swizzling, input2Swizzling};
        vpunnDPUWorkload.output_swizzling = {outputSwizzling};
        vpunnDPUWorkload.output_write_tiles = outputWriteTiles;
        vpunnDPUWorkload.weight_sparsity = weightsSparsityRatio;
        vpunnDPUWorkload.weight_sparsity_enabled = isWeightsSparsityEnabled;
        vpunnDPUWorkload.isi_strategy = isiStrategy;

        // TODO: Should RUNTIME_OVERHEAD_PER_WORKLOAD be added?
        auto cost = VPU::checkAndReturnCost(costModel->DPU(vpunnDPUWorkload), log, true);
        nceVariantCycles.push_back(cost);
    }

    VPUX_THROW_UNLESS(clusters.size() == 1, "Multicluster op not supported by this method - {0}", nceOp->getLoc());

    return nceVariantCycles;
}

size_t vpux::calculateNceCycles(VPUIP::NCEClusterTaskOp nceOp, const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                                VPU::ArchKind arch, vpux::Logger log, int64_t numDPU) {
    auto variantCostVec = calculateNceVariantCycles(nceOp, costModel, arch, log);

    return VPUNN::dpu_schedule(numDPU, variantCostVec);
}
