//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/cost_model_data.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Quant/QuantTypes.h>

using namespace vpux;

namespace {

ArrayRef<char> getCostModelData(VPU::ArchKind archKind, bool isFastModel) {
    switch (archKind) {
    case VPU::ArchKind::VPUX30XX:
    case VPU::ArchKind::VPUX311X:
        return makeArrayRef(VPU::COST_MODEL_2_0, VPU::COST_MODEL_2_0_SIZE);
    case VPU::ArchKind::VPUX37XX:
        if (isFastModel) {
            return makeArrayRef(VPU::COST_MODEL_2_7_FAST, VPU::COST_MODEL_2_7_FAST_SIZE);
        }
        return makeArrayRef(VPU::COST_MODEL_2_7, VPU::COST_MODEL_2_7_SIZE);
    default:
        VPUX_THROW("Unsupported VPU arch type: '{0}'", archKind);
    }
}

}  // namespace

std::shared_ptr<VPUNN::VPUCostModel> vpux::VPU::createCostModel(ArchKind arch) {
    // Track [E#70055]
    // TODO: Do not switch vpunn model to FAST temporarily, need to investigate the impact for workloads generation pass
    bool isFastModel = false;
    const auto costModelData = getCostModelData(arch, isFastModel);
    return std::make_shared<VPUNN::VPUCostModel>(costModelData.data(), costModelData.size(), false);
}

std::shared_ptr<VPUNN::VPULayerCostModel> vpux::VPU::createLayerCostModel(ArchKind arch, bool isFastModel) {
    // VPUNN provides two models - default and fast.
    // Currently use default model for workload generation. Ticket to explore moving to fast model [E#70055].
    // Currently use fast model for per layer evaluation in multi-cluster strategy selection
    const auto costModelData = getCostModelData(arch, isFastModel);
    return std::make_shared<VPUNN::VPULayerCostModel>(costModelData.data(), costModelData.size(), false);
}

///@brief Validate vpunn cost. If cost is not the defined error code then return it
/// Else print error code and return INVALID_COST (a uint32::max) to user.
/// Please report to E#80022 if any error code found in compilation log.
uint32_t vpux::VPU::checkAndReturnCost(const VPUNN::CyclesInterfaceType& cost, vpux::Logger log, bool beSilent) {
    if (VPUNN::Cycles::isErrorCode(cost)) {
        auto errorCode = VPUNN::Cycles::toErrorText(cost);
        if (beSilent) {
            log.trace("VPUNN error code {0} catched, code val {1}", errorCode, cost);
        } else {
            log.error("VPUNN error code {0} catched, code val {1}", errorCode, cost);
        }
        return VPU::INVALID_COST;
    }
    return cost;
}

///@brief Print vpunn config info
void vpux::VPU::printVPUNNLayerConfig(const VPUNN::DPULayer& layer, const VPUNN::VPULayerStrategy& strategy) {
    std::cout << "[VPUNN LOG] Layer config: " << layer << std::endl;
    std::cout << "[VPUNN LOG] Strategy config: " << strategy << std::endl;
}

///@brief Weights sparsity ratio basically is the math sparsity (the ratio of zero values) but considering the 16 Bytes
/// alignment for weights sets.
///@details A storage element is allocated to a weights set (ICxHxW), which has 16 Bytes alignment HW constraint.
/// Each weights set will be compressed to only include dense values and align to 16 B
/// And the total compressed_size stored in compressionSchemeAttr, which is calculated by sparsify-weights pass.
/// So ratio can be calculated by 1 - (compressed_size / total_size)
float vpux::VPU::getWeightsSparsityRatio(mlir::Value weights) {
    auto weightsGroupOp = weights.getDefiningOp<VPU::GroupSparseTensorOp>();
    VPUX_THROW_WHEN(weightsGroupOp == nullptr, "Expect a GroupSparseTensorOp for weights DefiningOp but got a nullptr");
    auto compressionSchemeAttr = weightsGroupOp.compression_schemeAttr();
    VPUX_THROW_WHEN(compressionSchemeAttr == nullptr, "compression_schemeAttr shouldn't be a nullptr");

    auto log = vpux::Logger("[calculate-sparstiy-ratio-vpunn]", LogLevel::None);
    log.trace("Calculate weights sparsity ratio for GroupSparseTensorOp {0}", weightsGroupOp->getLoc());
    auto weightsType = weights.getType().cast<vpux::NDTypeInterface>();
    auto originalSize = weightsType.getShape().totalSize();
    auto elemType = weightsType.getElementType();
    auto elemByteSize = vpux::getElemTypeSize(elemType).to<Byte>().count();
    auto originalAllocSize = originalSize * elemByteSize;
    auto compressedSize = compressionSchemeAttr.getAllocSize(elemType).count();

    // This check is to pass UNINIT.STACK.MUST check for "weightsSparsityRatio" in klocwork
    VPUX_THROW_WHEN(originalAllocSize == 0, "Denominator should be non-zero when doing division");
    float weightsSparsityRatio = 1.0 - (checked_cast<float>(compressedSize) / checked_cast<float>(originalAllocSize));
    log.trace(" Sparsity ratio: {0}", weightsSparsityRatio);
    VPUX_THROW_UNLESS(weightsSparsityRatio > 0.0 && weightsSparsityRatio <= 1.0,
                      "weightsSparsityRatio should be in range (0.0 , 1.0] however get {0}", weightsSparsityRatio);
    return weightsSparsityRatio;
}

VPUNN::VPUDevice vpux::VPU::getVPUDeviceType(VPU::ArchKind archKind) {
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

VPUNN::DataType vpux::VPU::getVPUNNElementType(mlir::Type type) {
    if (type.isBF16()) {
        return VPUNN::DataType::BFLOAT16;
    } else if (type.isF16()) {
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
    VPUX_THROW("Unsupported data type: '{0}'", type);
}

VPUNN::VPUTensor vpux::VPU::getVPUTensor(ShapeRef shape, mlir::Type elemType) {
    return VPUNN::VPUTensor(
            {
                    static_cast<unsigned int>(shape[Dims4D::Act::W]),
                    static_cast<unsigned int>(shape[Dims4D::Act::H]),
                    static_cast<unsigned int>(shape[Dims4D::Act::C]),
                    static_cast<unsigned int>(shape[Dims4D::Act::N]),
            },
            getVPUNNElementType(elemType));
}

/**
 * @param nTiles the number of CMX tiles
 * @param nDPUs Number of DPU per CMX tile
 * @param nSHVs the number of Act_Shave per CMX tiles
 */
VPUNN::VPULayerStrategy vpux::VPU::getVPULayerStrategy(VPU::MultiClusterStrategy strategy, size_t nDPUs, size_t nTiles,
                                                       size_t nSHVs, bool prefetching) {
    VPUNN::VPULayerStrategy VPUNNStrategy;
    VPUNNStrategy.nDPUs = static_cast<unsigned int>(nDPUs);
    VPUNNStrategy.nSHVs = static_cast<unsigned int>(nSHVs);
    VPUNNStrategy.nTiles = static_cast<unsigned int>(nTiles);
    VPUNNStrategy.prefetching = prefetching;

    switch (strategy) {
    case VPU::MultiClusterStrategy::SplitOverHeight:
    case VPU::MultiClusterStrategy::SplitOverHeightOverlapped:
    case VPU::MultiClusterStrategy::HKSwitch:
        VPUNNStrategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOH;
        return VPUNNStrategy;
    case VPU::MultiClusterStrategy::SplitOverKernel:
        VPUNNStrategy.tiling_strategy = VPUNN::VPUTilingStrategy::SOK;
        return VPUNNStrategy;
    case VPU::MultiClusterStrategy::Clustering:
        VPUNNStrategy.tiling_strategy = VPUNN::VPUTilingStrategy::NONE;
        return VPUNNStrategy;
    default:
        VPUX_THROW("Unsupported cluster-tiling strategy: '{0}'", strategy);
    }
}

VPUNN::DPULayer vpux::VPU::getDPULayer(const VPUIP::WorkloadCostParams& params) {
    VPUX_THROW_WHEN(params.kernelSize.size() < 2, "Kernel array size less than 2");
    const unsigned int KY = params.kernelSize[Dims4D::Kernel::Y.ind()];
    const unsigned int KX = params.kernelSize[Dims4D::Kernel::X.ind()];

    VPUX_THROW_WHEN(params.kernelStride.size() < 2, "Kernel stride array size less than 2");
    const unsigned int SY = params.kernelStride[Dims4D::Strides::Y.ind()];
    const unsigned int SX = params.kernelStride[Dims4D::Strides::X.ind()];

    const auto opType = getOperationType(params.nceTaskType);

    const auto outputTensor = VPU::getVPUTensor(params.outputShape, params.outDataType);
    const auto inputTensor = VPU::getVPUTensor(params.inputShape, params.inDataType);

    auto vpunnLayer = VPUNN::DPULayer(
            getVPUDeviceType(params.arch), opType, {inputTensor}, {outputTensor}, {KX, KY}, {SX, SY},
            {static_cast<unsigned int>(params.padInfo.top), static_cast<unsigned int>(params.padInfo.bottom),
             static_cast<unsigned int>(params.padInfo.left), static_cast<unsigned int>(params.padInfo.right)});
    vpunnLayer.set_weight_sparsity(params.isWeightsSparsityEnabled, params.weightsSparsityRatio);
    return vpunnLayer;
}

VPUIP::WorkloadCostParams vpux::VPU::getWorkloadCostParam(VPU::NCEOpInterface nceOp, VPU::ArchKind arch,
                                                          int64_t numDPU) {
    const auto inputType = nceOp->getOperand(0).getType().cast<NDTypeInterface>();
    const auto outputType = nceOp->getResult(0).getType().cast<NDTypeInterface>();
    const auto inElemType = inputType.getElementType();
    const auto outElemType = outputType.getElementType();

    const auto inputShape = inputType.getShape();
    const auto outputShape = outputType.getShape();

    const auto pads = nceOp.getPad();

    VPUIP::WorkloadCostParams params = {};
    params.inDataType = inElemType;
    params.outDataType = outElemType;
    params.numDPU = numDPU;
    params.arch = arch;
    params.fullInputShape = inputShape.raw();
    params.inputShape = inputShape.raw();
    params.outputShape = outputShape.raw();
    params.padInfo = VPU::toPadInfo(pads);
    params.kernelSize = nceOp.getKernelSize();
    params.kernelStride = nceOp.getStrides();
    params.weightsSparsityRatio = 0;
    params.isWeightsSparsityEnabled = false;

    // Considering weights sparsity. For CONV, DW_CONV ops
    const auto weights = nceOp.getWeightsOperand();
    if (weights != nullptr && weights.getType().isa<VPU::SparseTensorType>()) {
        params.weightsSparsityRatio = getWeightsSparsityRatio(weights);
        params.isWeightsSparsityEnabled = true;
    }

    llvm::TypeSwitch<mlir::Operation*, void>(nceOp.getOperation())
            .Case<VPU::NCEConvolutionOp>([&](VPU::NCEConvolutionOp) {
                const auto inOrder = inputType.getDimsOrder();
                const auto isCMajor = inOrder == DimsOrder::NCHW;
                params.nceTaskType = isCMajor ? VPUIP::NCETaskType::CMCONV : VPUIP::NCETaskType::CONV;
            })
            .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp) {
                params.nceTaskType = VPUIP::NCETaskType::DWCONV;
            })
            .Case<VPU::NCEMaxPoolOp>([&](VPU::NCEMaxPoolOp) {
                params.nceTaskType = VPUIP::NCETaskType::MAXPOOL;
            })
            .Case<VPU::NCEAveragePoolOp>([&](VPU::NCEAveragePoolOp) {
                params.nceTaskType = VPUIP::NCETaskType::AVEPOOL;
            })
            .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp) {
                params.nceTaskType = VPUIP::NCETaskType::ELTWISE;
            })
            .Case<VPU::NCEInterpolateOp>([&](VPU::NCEInterpolateOp) {
                params.nceTaskType = VPUIP::NCETaskType::CONV;
            })
            .Default([](mlir::Operation* op) {
                VPUX_THROW("Unsupported NCE operation '{0}' at '{1}'", op->getName(), op->getLoc());
            });
    return params;
}
