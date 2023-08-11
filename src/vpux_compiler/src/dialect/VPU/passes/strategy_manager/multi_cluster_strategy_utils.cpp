//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/strategy_manager.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include <unordered_map>

using namespace vpux;
using namespace VPU;

namespace {

enum class SpillingType { SPILL_WRITE, SPILL_READ };

VPUNN::VPUTensor getVPUNNTensorMultiCluster(ArrayRef<Shape> tensorShapes, mlir::Type dataType) {
    unsigned int totalSize =
            std::accumulate(tensorShapes.begin(), tensorShapes.end(), 0, [](unsigned int sum, Shape tensorShape) {
                return sum + static_cast<unsigned int>(tensorShape.totalSize());
            });
    return VPUNN::VPUTensor({totalSize, 1, 1, 1}, getVPUNNElementType(dataType));
}

double getSpillingCostForNonMultiCluster(vpux::NDTypeInterface tensorType, SpillingType /*spillingType*/,
                                         double ddrLatency, double ddrBandwidth, double /*cmxLatency*/,
                                         double /*cmxBandwidth*/) {
    // calculate the data byte size need copy from cmx to ddr or vice versa
    const auto totalSize = static_cast<double>(tensorType.getTotalAllocSize().count());
    return ddrLatency + totalSize / ddrBandwidth;
}

double getSpillingCostForDuplicated(vpux::NDTypeInterface tensorType, SpillingType /*spillingType*/, double ddrLatency,
                                    double ddrBandwidth, double /*cmxLatency*/, double /*cmxBandwidth*/) {
    auto distributedTensorType = tensorType.dyn_cast<DistributedTensorType>();
    VPUX_THROW_WHEN(distributedTensorType == nullptr, "Invalid type: {0}", tensorType);
    const auto totalSize = tensorType.getTotalAllocSize().count();
    return ddrLatency + totalSize / ddrBandwidth;
}

double getSpillingCostForDuplicated(vpux::NDTypeInterface tensorType, VPUNN::VPUDevice vpuDevice,
                                    const std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto shape = tensorType.getShape();
    auto elemType = tensorType.getElementType();
    auto vpuTensor = VPU::getVPUTensor(shape, elemType);
    return costModel->DMA(vpuDevice, vpuTensor, vpuTensor);
}

double getSpillingCostForSegmented(vpux::NDTypeInterface tensorType, SpillingType, double ddrLatency,
                                   double ddrBandwidth, double, double) {
    auto distributedTensorType = tensorType.dyn_cast<DistributedTensorType>();
    VPUX_THROW_WHEN(distributedTensorType == nullptr, "Invalid type: {0}", tensorType);
    auto shapes = distributedTensorType.getPerClusterComputeShapes();
    int64_t totalSize = 0;
    // sums up of all the sub tensors to get the total size
    for (auto& shape : shapes) {
        totalSize += shape.totalSize();
    }

    const auto elemSize = tensorType.getElemTypeSize();
    const auto byteSize = static_cast<int64_t>(CHAR_BIT);
    if (elemSize.count() < byteSize) {
        totalSize = vpux::divUp(totalSize, byteSize);
    } else {
        totalSize *= Byte(elemSize).count();
    }

    return ddrLatency + static_cast<double>(totalSize) / ddrBandwidth;
}

double getSpillingCostForSegmented(vpux::NDTypeInterface tensorType, VPUNN::VPUDevice vpuDevice,
                                   const std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto distributedTensorType = tensorType.dyn_cast<DistributedTensorType>();
    VPUX_THROW_WHEN(distributedTensorType == nullptr, "Invalid type: {0}", tensorType);
    auto shapes = distributedTensorType.getPerClusterComputeShapes();
    auto elemType = tensorType.getElementType();
    auto vpuTensor = getVPUNNTensorMultiCluster(shapes, elemType);
    return costModel->DMA(vpuDevice, vpuTensor, vpuTensor);
}

using GetSpillingCostCB = double (*)(vpux::NDTypeInterface, SpillingType, double ddrLatency, double ddrBandwidth,
                                     double cmxLatency, double cmxBandwidth);
const EnumMap<DistributionMode, GetSpillingCostCB> spillingCostMap{
        // using  DistributionMode::NONE for single clustering case
        {DistributionMode::NONE, getSpillingCostForNonMultiCluster},
        {DistributionMode::DUPLICATED, getSpillingCostForDuplicated},
        {DistributionMode::SEGMENTED, getSpillingCostForSegmented},
        {DistributionMode::OVERLAPPED, getSpillingCostForSegmented},
        {DistributionMode::MULTICASTED, getSpillingCostForDuplicated},
        {DistributionMode::DUPLICATED | DistributionMode::SEGMENTED, getSpillingCostForDuplicated},
        {DistributionMode::MULTICASTED | DistributionMode::SEGMENTED, getSpillingCostForDuplicated},
};

using GetSpillingCostOnVPUNN = double (*)(vpux::NDTypeInterface tensortType, VPUNN::VPUDevice vpuDevice,
                                          const std::shared_ptr<VPUNN::VPUCostModel>& costModel);
const EnumMap<DistributionMode, GetSpillingCostOnVPUNN> spillingCostMapVPUNN{
        // using  DistributionMode::NONE for single clustering case
        {DistributionMode::NONE, getSpillingCostForDuplicated},
        {DistributionMode::DUPLICATED, getSpillingCostForDuplicated},
        {DistributionMode::SEGMENTED, getSpillingCostForSegmented},
        {DistributionMode::OVERLAPPED, getSpillingCostForSegmented},
        {DistributionMode::MULTICASTED, getSpillingCostForDuplicated},
        {DistributionMode::DUPLICATED | DistributionMode::SEGMENTED, getSpillingCostForDuplicated},
        {DistributionMode::MULTICASTED | DistributionMode::SEGMENTED, getSpillingCostForDuplicated},
};

mlir::Value getInputFromClusteredOp(VPU::ClusteredOpInterface clusteredOp, mlir::Operation* parentOp) {
    for (auto operand : clusteredOp->getOperands()) {
        auto parent = operand.getDefiningOp();
        if (parent == parentOp) {
            return operand;
        }
        while (mlir::isa_and_nonnull<VPU::ShapeCastOp, VPU::QuantizeCastOp>(parent)) {
            // propagate cast ops
            parent = parent->getOperand(0).getDefiningOp();
            if (parent == parentOp) {
                return operand;
            }
        }
    }

    VPUX_THROW("Cannot find input from op: {0}, parent op: {1}", clusteredOp, parentOp);
}

bool hasUserMVN(VPU::ClusteredOpInterface clusteredOp) {
    if (!clusteredOp->getOperand(0).isa<mlir::BlockArgument>() &&
        mlir::isa<VPU::MVNOp>(clusteredOp->getOperand(0).getDefiningOp())) {
        // MVN producer
        return true;
    }
    for (auto* user : clusteredOp->getResult(0).getUsers()) {
        // MVN consumer
        if (mlir::isa<VPU::MVNOp>(user)) {
            return true;
        }
    }
    return false;
}

bool isAlignmentCompatible(VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface srcType,
                           vpux::NDTypeInterface dstType) {
    const auto srcTypeInterface = srcType.dyn_cast<VPU::DistributedTypeInterface>();
    const auto dstTypeInterface = dstType.dyn_cast<VPU::DistributedTypeInterface>();
    if (srcTypeInterface != nullptr && dstTypeInterface != nullptr) {
        const auto distributedSrcTypes = srcTypeInterface.getDistributedTypes();
        const auto distributedDstTypes = dstTypeInterface.getDistributedTypes();
        if (distributedSrcTypes.size() != distributedDstTypes.size()) {
            return false;
        }
        if (distributedSrcTypes.empty()) {
            return false;
        }
        const auto distSrcDataType = distributedSrcTypes.front().cast<VPU::DistributedTensorType>();
        const auto distDstDataType = distributedDstTypes.front().cast<VPU::DistributedTensorType>();
        if ((distSrcDataType.getDistribution().mode().getValue() == DistributionMode::SEGMENTED) &&
            (distDstDataType.getDistribution().mode().getValue() == DistributionMode::SEGMENTED) &&
            (distSrcDataType.getDistribution().num_tiles() == distDstDataType.getDistribution().num_tiles())) {
            const auto newDistributedOutputType = adjustOutputAlignmentForSOH(clusteredOp, dstType);
            if (newDistributedOutputType.hasValue()) {
                const auto newDistributedOutputTypeInterface =
                        newDistributedOutputType.getValue().cast<VPU::DistributedTypeInterface>();
                const auto newDistributedDstTypes = newDistributedOutputTypeInterface.getDistributedTypes();
                for (auto p : zip(newDistributedDstTypes, distributedDstTypes)) {
                    const auto newDistributedDstType = std::get<0>(p).cast<VPU::DistributedTensorType>();
                    const auto distributedDstType = std::get<1>(p).cast<VPU::DistributedTensorType>();
                    if (isDistributedCastCompatible(newDistributedDstType, distributedDstType).failed()) {
                        return false;
                    }
                }
                return true;
            }
        }
    }

    return false;
}

bool isTargetTensorTypeCompatible(vpux::NDTypeInterface srcType, vpux::NDTypeInterface dstType) {
    const auto srcTypeInterface = srcType.dyn_cast<VPU::DistributedTypeInterface>();
    const auto dstTypeInterface = dstType.dyn_cast<VPU::DistributedTypeInterface>();
    const auto srcTypeIsDistributed = srcTypeInterface != nullptr;
    const auto dstTypeIsDistributed = dstTypeInterface != nullptr;
    if (srcTypeIsDistributed ^ dstTypeIsDistributed) {
        return false;
    }
    if (srcTypeIsDistributed && dstTypeIsDistributed) {
        const auto srcContainsDistTypes = srcTypeInterface.containsDistributedTypes();
        const auto dstContainsDistTypes = dstTypeInterface.containsDistributedTypes();
        if (srcContainsDistTypes ^ dstContainsDistTypes) {
            return false;
        }
        if (srcContainsDistTypes && dstContainsDistTypes) {
            const auto distributedSrcTypes = srcTypeInterface.getDistributedTypes();
            const auto distributedDstTypes = dstTypeInterface.getDistributedTypes();
            if (distributedSrcTypes.size() != distributedDstTypes.size()) {
                return false;
            }
            for (auto p : zip(distributedSrcTypes, distributedDstTypes)) {
                const auto distributedSrcType = std::get<0>(p).cast<VPU::DistributedTensorType>();
                const auto distributedDstType = std::get<1>(p).cast<VPU::DistributedTensorType>();
                if (isDistributedCastCompatible(distributedSrcType, distributedDstType).failed()) {
                    return false;
                }
            }
        }
    }
    return true;
}
}  // namespace

LayerCostModel::LayerCostModel(mlir::func::FuncOp func, Logger log): _func(func), _log(log) {
    auto module = func->getParentOfType<mlir::ModuleOp>();

    if (auto nceOp = IE::getAvailableExecutor(module, ExecutorKind::NCE)) {
        auto dpuExec = nceOp.getSubExecutor(VPU::ExecutorKindAttr::get(module->getContext(), VPU::ExecutorKind::DPU));
        _NCEFrequency = nceOp.getProcessorFrequency().getValueAsDouble();
        _numClusters = nceOp.count();
        _numDPUs = dpuExec.count();
        _NCEThroughput = getNCEThroughput(VPU::getArch(nceOp));
        _DMABandwidth = getDMABandwidth(VPU::getArch(nceOp));
    }
    if (auto shaveActExec = IE::getAvailableExecutor(module, ExecutorKind::SHAVE_ACT)) {
        _numShaveActs = shaveActExec.count();
    }

    _arch = VPU::getArch(module);
    _vpuDeviceType = VPU::getVPUDeviceType(_arch);
    _layerCostModel = VPU::createLayerCostModel(_arch);
}

vpux::NDTypeInterface LayerCostModel::getNormalInputType(VPU::ClusteredOpInterface origOp,
                                                         mlir::Operation* parentOp) const {
    auto input = getInputFromClusteredOp(origOp, parentOp);
    return input.getType().dyn_cast<vpux::NDTypeInterface>();
}

vpux::NDTypeInterface LayerCostModel::getNormalOutputType(VPU::ClusteredOpInterface origOp) const {
    auto output = origOp->getResult(0);
    return output.getType().dyn_cast<vpux::NDTypeInterface>();
}

VPU::DistributedTypeInterface LayerCostModel::getDistributedInputType(
        VPU::ClusteredOpInterface origOp, mlir::Operation* parentOp,
        VPU::MultiClusterStrategy specifiedStrategy) const {
    auto input = getInputFromClusteredOp(origOp, parentOp);
    auto numClustersAttr = VPU::getOptimalNumClusters(
            origOp, origOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C],
            specifiedStrategy);
    return getDistributedActivationTypeFromOp(origOp, input.getType(), numClustersAttr, specifiedStrategy);
}

VPU::DistributedTypeInterface LayerCostModel::getDistributedInputType(VPU::ClusteredOpInterface origOp,
                                                                      mlir::Operation* parentOp,
                                                                      VPU::MultiClusterStrategy specifiedStrategy,
                                                                      mlir::ArrayAttr customAlignment) const {
    auto input = getInputFromClusteredOp(origOp, parentOp);
    auto numClustersAttr = VPU::getOptimalNumClusters(
            origOp, origOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C],
            specifiedStrategy);
    return getDistributedActivationTypeFromOp(origOp, input.getType(), numClustersAttr, specifiedStrategy,
                                              customAlignment);
}

VPU::DistributedTypeInterface LayerCostModel::getDistributedOutputType(
        VPU::ClusteredOpInterface origOp, VPU::MultiClusterStrategy specifiedStrategy) const {
    auto numClustersAttr = VPU::getOptimalNumClusters(
            origOp, origOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C],
            specifiedStrategy);
    return VPU::getDistributedOutputTypeFromOp(origOp, origOp->getResult(0).getType(), numClustersAttr,
                                               specifiedStrategy);
}

/*
 * Get the spilling cost
 * srcTensorType is the output of parent op (current op)
 * dstTensorType is the input of child op
 * return spilling write cost and spilling read cost
 */
LayerCostModel::SpillingCost LayerCostModel::getSpillingCost(vpux::NDTypeInterface srcTensorType,
                                                             vpux::NDTypeInterface dstTensorType,
                                                             VPU::ClusteredOpInterface parentOp,
                                                             VPU::ClusteredOpInterface userOp) const {
    // Concat is on DDR memory if there's spilling. So we don't need copy from CMX to DDR if Concat is parent. Also we
    // don't need copy from DDR to CMX if Concat is user.
    if (mlir::isa<VPU::ConcatOp>(parentOp)) {
        return {0.0, getSpillingReadCost(dstTensorType)};
    }

    if (mlir::isa<VPU::ConcatOp>(userOp)) {
        return {getSpillingWriteCost(srcTensorType), 0.0};
    }

    return {getSpillingWriteCost(srcTensorType), getSpillingReadCost(dstTensorType)};
}

double LayerCostModel::getSpillingReadCost(vpux::NDTypeInterface srcTensorType) const {
    auto distributedSrcType = srcTensorType.dyn_cast<DistributedTensorType>();
    auto srcMode = distributedSrcType != nullptr ? distributedSrcType.getDistribution().mode().getValue()
                                                 : VPU::DistributionMode::NONE;

    if (_arch == ArchKind::VPUX37XX) {
        auto spillingReadCostFunc = spillingCostMapVPUNN.at(srcMode);
        return spillingReadCostFunc(srcTensorType, _vpuDeviceType, _layerCostModel);
    }
    auto spillingReadCostFunc = spillingCostMap.at(srcMode);
    return spillingReadCostFunc(srcTensorType, SpillingType::SPILL_READ, _DDRLatency, _DMABandwidth, _CMXLatency,
                                _CMXMulticastBandwidth);
}

double LayerCostModel::getSpillingWriteCost(vpux::NDTypeInterface srcTensorType) const {
    auto distributedSrcType = srcTensorType.dyn_cast<DistributedTensorType>();
    auto srcMode = distributedSrcType != nullptr ? distributedSrcType.getDistribution().mode().getValue()
                                                 : VPU::DistributionMode::NONE;

    if (_arch == ArchKind::VPUX37XX) {
        auto spillingWriteCostFunc = spillingCostMapVPUNN.at(srcMode);
        return spillingWriteCostFunc(srcTensorType, _vpuDeviceType, _layerCostModel);
    }
    auto spillingWriteCostFunc = spillingCostMap.at(srcMode);
    return spillingWriteCostFunc(srcTensorType, SpillingType::SPILL_WRITE, _DDRLatency, _DMABandwidth, _CMXLatency,
                                 _CMXMulticastBandwidth);
}

// The function computes the actual output tensor volume (i.e. computation that is performed)
// given the stratey and the MPE mode
double LayerCostModel::calculateMPEVolume(VPU::MPEMode mpeMode, Shape shape) const {
    int64_t mpeHeight;
    int64_t mpeWidth;
    switch (mpeMode) {
    case VPU::MPEMode::VECTOR: {
        mpeHeight = 1;
        mpeWidth = 16;
        break;
    }
    case VPU::MPEMode::VECTOR_FP16: {
        mpeHeight = 1;
        mpeWidth = 4;
        break;
    }
    case VPU::MPEMode::MATRIX:
    // These different mpe modes on VPUX37XX have impact on the reuse of activation and weights. We can't estimate reuse
    // cost with current cost equation. In the future we will integrate VPUNN to estimate the cost.
    case VPU::MPEMode::CUBOID_4x16:
    case VPU::MPEMode::CUBOID_8x16:
    case VPU::MPEMode::CUBOID_16x16: {
        mpeHeight = 4;
        mpeWidth = 4;
        break;
    }
    default:
        VPUX_THROW("Unsupported mpeMode '{0}'", mpeMode);
    }

    return static_cast<double>(_numDPUs * divUp((mpeHeight * divUp(shape[Dims4D::Act::H], mpeHeight) * mpeWidth *
                                                 divUp(shape[Dims4D::Act::W], mpeWidth) * _numChannelAlignment *
                                                 divUp(shape[Dims4D::Act::C], _numChannelAlignment)),
                                                _numDPUs));
}

// The efficiency calculation that is being performed here can be described as follows.
// A ratio of the real output tensor volume to the actual computation that occurs on the
// hardware for each MPE Mode 4x4x16 and 16x1x16 is computed and the maximum is selected.
double LayerCostModel::computeSplitEfficiency(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp.checkStrategyCompatibility(strategy) == true,
                      "Unsupported multi-cluster strategy '{0}' for layer type '{1}'", strategy, nceOp->getName());
    auto numClusters = getOptimalNumClusters(
            clusteredOp, nceOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C],
            strategy);
    const auto distributedOutputTensorType = VPU::getDistributedOutputTypeFromOp(
            clusteredOp, nceOp->getResult(0).getType().cast<vpux::NDTypeInterface>(), numClusters, strategy);

    VPUX_THROW_UNLESS(distributedOutputTensorType.containsDistributedTypes(), "Missing output distributed types");
    const auto distributedOutputDataType =
            distributedOutputTensorType.getDistributedTypes().front().cast<VPU::DistributedTensorType>();
    const auto perClusterShape = distributedOutputDataType.getLargestCompactShape();
    const auto perClusterOutputTensorVolume =
            perClusterShape[Dims4D::Act::H] * perClusterShape[Dims4D::Act::W] * perClusterShape[Dims4D::Act::C];

    const auto arch = VPU::getArch(nceOp);

    // VPUX30XX and VPUX37XX have different kinds of MPE mode
    if (arch == ArchKind::VPUX37XX) {
        return std::max(std::max(static_cast<double>(perClusterOutputTensorVolume) /
                                         calculateMPEVolume(VPU::MPEMode::CUBOID_4x16, perClusterShape),
                                 static_cast<double>(perClusterOutputTensorVolume) /
                                         calculateMPEVolume(VPU::MPEMode::CUBOID_8x16, perClusterShape)),
                        static_cast<double>(perClusterOutputTensorVolume) /
                                calculateMPEVolume(VPU::MPEMode::CUBOID_16x16, perClusterShape));
    } else {
        return std::max(static_cast<double>(perClusterOutputTensorVolume) /
                                calculateMPEVolume(VPU::MPEMode::MATRIX, perClusterShape),
                        static_cast<double>(perClusterOutputTensorVolume) /
                                calculateMPEVolume(VPU::MPEMode::VECTOR, perClusterShape));
    }
}

// Returns the duration in cycles for the execution of a NCE task
double LayerCostModel::clusterComputeTime(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp.checkStrategyCompatibility(strategy) == true,
                      "Unsupported multi-cluster strategy '{0}' for layer type '{1}'", strategy, nceOp->getName());

    double clusterOpsPerCycle = _NCEThroughput / _NCEFrequency / _numClusters;
    double clusterEff = computeSplitEfficiency(nceOp, strategy);
    auto largestClusterOutShape = getLargestClusterOutputShape(clusteredOp, strategy);

    auto kernelSize = nceOp.getKernelSize();
    auto op = nceOp.getOperation();
    int64_t baseKernelCost = kernelSize[Dims4D::Kernel::Y.ind()] * kernelSize[Dims4D::Kernel::X.ind()];
    if (mlir::isa<VPU::NCEConvolutionOp, VPU::NCECompressConvolutionOp, VPU::NCEInterpolateOp>(op)) {
        int64_t IC = getShape(
                op->getOperand(0))[Dims4D::Act::C];  // Get input channel (already channel-alignment in previous pass)
        baseKernelCost = IC * baseKernelCost;

    } else if (mlir::isa<VPU::NCEEltwiseOp>(op)) {
        baseKernelCost = 1;
    } else if (!mlir::isa<VPU::NCEMaxPoolOp>(op) && !mlir::isa<VPU::NCEAveragePoolOp>(op) &&
               !mlir::isa<VPU::NCEDepthConvolutionOp>(op)) {
        VPUX_THROW("Invalid NCE operation type: '{0}'", op->getName());
    }

    // Here calculating the total basic operation number for the largest cluster output
    // And also we can reduce formula like:
    // basicOperationVolume = clusterOutShapeSize * baseKernelCost / Efficiency
    //                      = clusterOutShapeSize * baseKernelCost / (clusterOutShapeSize / MPEVolume) =
    //                      = MPEVolume * baseKernelCost
    // So that MPEVolume * baseKernelCost is the final result, and then we can divide frequency to get final cycles
    double basicOperationVolume =
            (static_cast<double>(largestClusterOutShape.totalSize() * baseKernelCost)) / clusterEff;
    double clusterComputeCycles = basicOperationVolume / clusterOpsPerCycle;
    return clusterComputeCycles;
}

/// @brief Returns total number of cycles required to weights DMA and CMX broadcast
/// for a layer with given strategy
/// @details Data transferring cost is modeled as (latency + size / bandwidth)
double LayerCostModel::totalDMATime(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp.checkStrategyCompatibility(strategy) == true,
                      "Unsupported multi-cluster strategy '{0}' for layer type '{1}'", strategy, nceOp->getName());
    double totalActivationWindowCycles = 0.0;
    double totalWeightCycles = 0.0;
    double totalWeightsTableCycles = 0.0;
    double pwlTableCycles = 0.0;
    double outputCycles = 0.0;
    const auto op = nceOp.getOperation();
    const auto inOrder = DimsOrder::fromValue(op->getOperand(0));
    const bool isCMajor = inOrder == DimsOrder::NCHW;
    const int64_t IC = (mlir::isa<VPU::NCEConvolutionOp, VPU::NCECompressConvolutionOp, VPU::NCEInterpolateOp>(op))
                               ? getShape(op->getOperand(0))[Dims4D::Act::C]
                               : 1;
    const int64_t OC = getShape(op->getResult(0))[Dims4D::Act::C];
    const auto kernelSize = nceOp.getKernelSize();
    auto numClustersAttr = VPU::getOptimalNumClusters(clusteredOp, OC, strategy);

    /// Weights cost
    /// Weights and weightTable are Segmented mode under SOK (only including ddr -> cmx cost),
    /// SOK may use less clusters to avoid alignment
    /// So it's not proper to estimate total weightsSize by "clusterWeightsSize * _numClusters" simply
    /// Using distributed tensor for SOK to get accurate total size
    if (mlir::isa<VPU::NCEConvolutionOp, VPU::NCEDepthConvolutionOp, VPU::NCECompressConvolutionOp,
                  VPU::NCEInterpolateOp>(op)) {
        auto weights = op->getOperand(1);

        SmallVector<int64_t> numElemsPerOC;
        int64_t weightSetAlignment = _cmxAddressAlignment;
        if (auto sparseWeightsType = weights.getType().dyn_cast<VPU::SparseTensorType>()) {
            VPU::CompressionSchemeAttr compressionScheme = sparseWeightsType.getCompressionScheme();
            if (compressionScheme != nullptr && compressionScheme.getAxis() != nullptr) {
                auto axis = compressionScheme.getAxis().getInt();
                VPUX_THROW_UNLESS(axis == Dims4D::Filter::OC.ind(),
                                  "SplitOverK is only compatible with compression over OC");
                numElemsPerOC = to_small_vector(compressionScheme.getNumElems().getValues<int64_t>());
                if (compressionScheme.getAlignment() != nullptr) {
                    weightSetAlignment = compressionScheme.getAlignment().getInt();
                }

                auto weightsShape = sparseWeightsType.getShape();
                VPUX_THROW_UNLESS(
                        static_cast<int64_t>(numElemsPerOC.size()) == weightsShape[Dims4D::Filter::OC],
                        "Different number of output channels {0} compared to compression scheme with {1} elements",
                        weightsShape[Dims4D::Filter::OC], numElemsPerOC.size());
            }
        }

        if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
            int64_t totalWeightsSize = 0;
            auto distributedWeightsTensorType =
                    VPU::getDistributedFilterTypeFromOp(nceOp, weights.getType(), numClustersAttr, strategy);

            for (auto type : distributedWeightsTensorType.getDistributedTypes() | indexed) {
                auto distributedWeightsType = type.value().cast<VPU::DistributedTensorType>();
                auto tiledWeightsShapes = distributedWeightsType.getPerClusterMemoryShapes();
                auto tiledWeightsOffsets = distributedWeightsType.getPerClusterMemoryShapeOffsets();

                const double elemByteSize =
                        static_cast<double>(getElemTypeSize(distributedWeightsType).count()) / CHAR_BIT;

                int64_t weightsByteSize = 0;

                if (type.index() == 0 && !numElemsPerOC.empty()) {
                    for (auto p : zip(tiledWeightsShapes, tiledWeightsOffsets)) {
                        const auto tileShape = std::get<0>(p);
                        const auto tileOffsets = std::get<1>(p);
                        const auto startOC = tileOffsets[Dims4D::Filter::OC];
                        const auto endOC = startOC + tileShape[Dims4D::Filter::OC];
                        for (auto idx = startOC; idx < endOC; ++idx) {
                            const auto numElems = *(numElemsPerOC.begin() + idx);
                            const auto tileByteSize =
                                    static_cast<int64_t>(static_cast<double>(numElems) * elemByteSize);
                            const auto alignedTileByteSize =
                                    vpux::alignValUp<int64_t>(tileByteSize, weightSetAlignment);
                            weightsByteSize += alignedTileByteSize;
                        }
                    }
                } else {
                    for (auto& clusterWeightsShape : tiledWeightsShapes) {
                        const auto tileByteSize = static_cast<double>(clusterWeightsShape.totalSize()) * elemByteSize;
                        weightsByteSize += static_cast<int64_t>(tileByteSize);
                    }
                }

                totalWeightsSize += weightsByteSize;
            }

            const double weightCycles = static_cast<double>(totalWeightsSize) / _DMABandwidth;
            totalWeightCycles = _DDRLatency + weightCycles;
        } else {
            // Duplicated mode in other strategies has ddr->cmx cost
            // The weight set size (IC * KX * KY * BytesPerElement) needs to be aligned to 16B for kernels

            auto ndWeightsType = weights.getType().cast<vpux::NDTypeInterface>();
            const double elemByteSize = static_cast<double>(ndWeightsType.getElemTypeSize().count()) / CHAR_BIT;

            const int64_t weightSetSize =
                    IC * kernelSize[Dims4D::Kernel::Y.ind()] * kernelSize[Dims4D::Kernel::X.ind()];

            int64_t clusterWeightsSize = 0;
            if (!numElemsPerOC.empty()) {
                for (auto numElems : numElemsPerOC) {
                    clusterWeightsSize += alignValUp<int64_t>(numElems * elemByteSize, _cmxAddressAlignment);
                }
            } else {
                clusterWeightsSize += (OC * alignValUp<int64_t>(weightSetSize * elemByteSize, _cmxAddressAlignment));
            }

            if (weights.getType().isa<VPU::SparseTensorType>()) {
                const int64_t weightSetBitAlignment = 128;
                const int64_t sparsityMapSize = (OC * alignValUp<int64_t>(weightSetSize, weightSetBitAlignment));
                const int64_t sparsityMapByteSize = sparsityMapSize / CHAR_BIT;
                clusterWeightsSize += sparsityMapByteSize;
            }

            const double weightCycles = static_cast<double>(clusterWeightsSize) / _DMABandwidth;
            totalWeightCycles = _DDRLatency + weightCycles;
        }
    }

    /// WeightsTable cost
    /// WeightTable has OC entries, each entry includes sparsity/weights pointer, bias and mult/shfit quantized
    /// params. The total size for each entry is 16 Bytes
    if (mlir::isa<VPU::NCEConvolutionOp, VPU::NCEDepthConvolutionOp, VPU::NCEMaxPoolOp, VPU::NCECompressConvolutionOp,
                  VPU::NCEInterpolateOp>(op)) {
        auto largestClusterOutShape = getLargestClusterOutputShape(clusteredOp, strategy);
        int64_t alignedClusterOutChannels = largestClusterOutShape[Dims4D::Act::C];
        int64_t clusterWeightTableSize = NCEInvariant::getWeightsTableSize(alignedClusterOutChannels).count();

        if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
            totalWeightsTableCycles =
                    _DDRLatency +
                    static_cast<double>(clusterWeightTableSize * numClustersAttr.getInt()) / _DMABandwidth;
        } else {
            totalWeightsTableCycles = _DDRLatency + (static_cast<double>(clusterWeightTableSize) / _DMABandwidth);
        }
    }

    /// ActivationWindow cost
    /// It's always duplicated mode and only dwconv , cmconv and maxpool own it
    int64_t activationWindowSize = 0;
    if (mlir::isa<VPU::NCEMaxPoolOp>(op) || mlir::isa<VPU::NCEDepthConvolutionOp>(op) ||
        (mlir::isa<VPU::NCEConvolutionOp>(op) && isCMajor)) {
        const auto SX = Shape(nceOp.getStrides())[Dims4D::Strides::X];
        const auto inputElemType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>().getElementType();
        auto sparsityMode = VPU::NCESparsity::Mode::POOL;
        if (mlir::isa<VPU::NCEConvolutionOp>(op)) {
            sparsityMode = VPU::NCESparsity::Mode::CM_CONV;
        } else if (mlir::isa<VPU::NCEDepthConvolutionOp>(op)) {
            sparsityMode = VPU::NCESparsity::Mode::DW_CONV;
        }

        activationWindowSize =
                VPU::NCESparsity::getActivationWindowSize(sparsityMode, Shape(kernelSize), SX, inputElemType, IC);
    }
    totalActivationWindowCycles = _DDRLatency + (static_cast<double>(activationWindowSize) / _DMABandwidth);

    /// PWLTable cost
    /// DistributionMode is Duplicated, at most we have 32 instructions (8 segments for pwl),
    /// so size is 32 x INT32 = 128 Bytes
    if (auto origOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(op)) {
        if (origOp.instructionListTable()) {
            pwlTableCycles = _DDRLatency + (128.0 / _DMABandwidth);
        }
    } else if (auto origOp = mlir::dyn_cast<VPU::NCEDepthConvolutionOp>(op)) {
        if (origOp.instructionListTable()) {
            pwlTableCycles = _DDRLatency + (128.0 / _DMABandwidth);
        }
    }

    // Total cost for single layer
    return totalActivationWindowCycles + totalWeightCycles + totalWeightsTableCycles + pwlTableCycles + outputCycles;
}

/// @brief A switcher to select time-cost or efficiency-cost for greedy strategy assignment
/// @details Time-cost includes an extra input spilling cost to be more accurate
double LayerCostModel::getNCELayerCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy,
                                       bool useTimeBasedCost) const {
    if (!useTimeBasedCost) {
        return getEfficiencyCost(nceOp, strategy);
    }

    auto basicDPUandDMACost = getDPUandDMATimeCost(nceOp, strategy);
    return basicDPUandDMACost;
}

/// @brief Time-cost : return the shave computation time (cycles)
/// @details use vpunn cost model to get the shave cost of sw layer
double LayerCostModel::getSWLayerCost(VPU::SWOpInterface swOp, VPU::MultiClusterStrategy strategy) const {
    auto inputType = swOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    auto outputType = swOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto inputTensor = VPU::getVPUTensor(inputType.getShape(), inputType.getElementType());
    auto outputTensor = VPU::getVPUTensor(outputType.getShape(), outputType.getElementType());
    std::shared_ptr<VPUNN::SWOperation> vpunnLayer;
    llvm::TypeSwitch<mlir::Operation*, void>(swOp.getOperation())
            .Case<VPU::TanhOp>([&](VPU::TanhOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVTanh>(getVPUDeviceType(_arch), inputTensor, outputTensor);
            })
            .Case<VPU::MVNOp>([&](VPU::MVNOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVMVN>(getVPUDeviceType(_arch), inputTensor, outputTensor);
            })
            .Case<VPU::SoftMaxOp>([&](VPU::SoftMaxOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVSoftmax>(getVPUDeviceType(_arch), inputTensor, outputTensor);
            })
            .Case<VPU::SwishOp>([&](VPU::SwishOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVSwish>(getVPUDeviceType(_arch), inputTensor, outputTensor);
            })
            .Case<VPU::HSwishOp>([&](VPU::HSwishOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVHardSwish>(getVPUDeviceType(_arch), inputTensor, outputTensor);
            })
            .Default([&](mlir::Operation* op) {
                VPUX_THROW("SW op {0} has no VPUNN support", op->getName());
            });
    auto vpunnStrategy = VPU::getVPULayerStrategy(strategy, _numDPUs, _numClusters, _numShaveActs, false);
    return _layerCostModel->Layer(*vpunnLayer, vpunnStrategy);
}

/// @brief get computation cost
/// @details Time-cost includes an extra input spilling cost to be more accurate
double LayerCostModel::getLayerCost(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy,
                                    bool useTimeBasedCost) const {
    if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(clusteredOp.getOperation())) {
        return getNCELayerCost(nceOp, strategy, useTimeBasedCost);
    } else if (auto swOp = mlir::dyn_cast<VPU::SWOpInterface>(clusteredOp.getOperation())) {
        return getSWLayerCost(swOp, strategy);
    } else if (mlir::isa<VPU::ConcatOp>(clusteredOp.getOperation())) {
        // Concat has no computation cost
        return 0.0;
    } else {
        VPUX_THROW("Unsupported op type {0} at {1}", clusteredOp->getName(), clusteredOp->getLoc());
    }
}

/// @brief Time-cost : return a sum of layer DPU time and weights DMA time (cycles)
/// @details DPU time calculation also considers the impact of workloads split efficiency
double LayerCostModel::getDPUandDMATimeCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    if (_arch == ArchKind::VPUX37XX) {
        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
        VPUX_THROW_WHEN(clusteredOp == nullptr, "NCE op {0} at {1} should be a clustered op", nceOp->getName(),
                        nceOp.getLoc());
        const auto costParam = VPU::getWorkloadCostParam(nceOp, _arch, _numDPUs);
        SmallVector<VPUNN::DPULayer> vpunnLayers{VPU::getDPULayer(costParam)};

        // Check CMX memory as VPUNN works with layer which fits CMX memory
        // If not, tiling big layer to fit into CMX
        auto layerStrategyChecker = LayerStrategyCheckerFactory::instance().get(clusteredOp->getName());
        if (!(layerStrategyChecker->doesLayerFitIntoCMX(clusteredOp, strategy))) {
            _log.trace("Tiling op {0} to fit into cmx before passing to VPUNN Layer API", nceOp.getLoc());
            auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(nceOp.getOperation());
            VPUX_THROW_WHEN(tilingOp == nullptr, "NCE op {0} at {1} should be a tiling op", nceOp->getName(),
                            nceOp.getLoc());

            OutputTiling outTiles;
            // Set customized strategy to the op to get corresponding output tiles when tiling
            // Save and restore original strategy if needed
            auto origStrategy = clusteredOp.getMultiClusterStrategyAttr();
            clusteredOp.setMultiClusterStrategyAttr(strategy);
            outTiles = getLayerTilingStrategy(tilingOp, false, _log);
            if (origStrategy.hasValue()) {
                clusteredOp.setMultiClusterStrategyAttr(origStrategy.getValue());
            } else {
                clusteredOp->removeAttr(multiClusterStrategy);
            }

            auto tilingVPUNNLayer = [&](const VPUNN::DPULayer& vpunnLayer,
                                        const OutputTiling& outTiles) -> SmallVector<VPUNN::DPULayer> {
                SmallVector<VPUNN::DPULayer> vpunnLayers;
                for (auto& outTile : outTiles) {
                    vpunnLayers.push_back(vpunnLayer);
                    auto inTiles = tilingOp.backInferTileInfo(outTile, _log);
                    auto& inputTile = inTiles.tiles.front();
                    auto inPad = inTiles.pads;
                    vpunnLayers.back().inputs = {getVPUTensor(inputTile.shape, costParam.inDataType)};
                    vpunnLayers.back().outputs = {getVPUTensor(outTile.shape, costParam.outDataType)};
                    if (inPad.hasValue()) {
                        vpunnLayers.back().padding = {
                                static_cast<unsigned int>(inPad->top), static_cast<unsigned int>(inPad->bottom),
                                static_cast<unsigned int>(inPad->left), static_cast<unsigned int>(inPad->right)};
                    }
                }
                return vpunnLayers;
            };
            // Return tiled vpunn layers
            vpunnLayers = tilingVPUNNLayer(vpunnLayers[0], outTiles);
        }

        auto vpunnStrategy = VPU::getVPULayerStrategy(strategy, _numDPUs, _numClusters);

        _log.trace("Start calculating vpunn cost for Op {0} with strategy {1}", nceOp.getLoc(), strategy);
        double vpunnCost = 0;
        for (auto& vpunnLayer : vpunnLayers) {
            uint32_t cost = checkAndReturnCost(_layerCostModel->Layer(vpunnLayer, vpunnStrategy), _log);
            if (cost == VPU::INVALID_COST) {
                printVPUNNLayerConfig(vpunnLayer, vpunnStrategy);
                return static_cast<double>(cost);
            }
            _log.trace(" VPUNN layer cost {0}", cost);
            vpunnCost += static_cast<double>(cost);
        }
        _log.trace("VPUNN total layer cost {0}", vpunnCost);
        return vpunnCost;
    }

    // For KMB
    return clusterComputeTime(nceOp, strategy) + totalDMATime(nceOp, strategy);
}

///@brief Effi-cost : A simple cost considering DPU computing efficiency
double LayerCostModel::getEfficiencyCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    return 1.0 / computeSplitEfficiency(nceOp, strategy);
}

bool LayerCostModel::hasMultiClusterStrategy(mlir::Operation* op) const {
    if (auto clusteringOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op)) {
        return clusteringOp.getMultiClusterStrategyAttr().hasValue();
    }

    return false;
}

VPU::MultiClusterStrategy LayerCostModel::getMultiClusterStrategyValue(VPU::ClusteredOpInterface clusteredOp) const {
    auto strategy = clusteredOp.getMultiClusterStrategyAttr();
    if (!strategy.hasValue()) {
        VPUX_THROW("NCE operation {0} doesn't have multiClusterStrategy attribute", clusteredOp->getLoc());
    }

    return strategy.getValue();
}

/// Anywhere if you need to judge spilling existing, please call me!
/// srcTensorType is the output of parent origOp
/// dstTensorType is the input of child NCE op
bool LayerCostModel::hasSpilling(VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface srcTensorType,
                                 vpux::NDTypeInterface dstTensorType) const {
    if (isTargetTensorTypeCompatible(srcTensorType, dstTensorType) ||
        isAlignmentCompatible(clusteredOp, srcTensorType, dstTensorType)) {
        return false;
    }
    return true;
}

std::pair<vpux::NDTypeInterface, vpux::NDTypeInterface> LayerCostModel::getDistributionTypesWithStrategy(
        VPU::ClusteredOpInterface parentOp, VPU::MultiClusterStrategy parentStrategy, VPU::ClusteredOpInterface userOp,
        VPU::MultiClusterStrategy userStrategy) const {
    // Set the custom strategy to the op to get the accurate distributed type
    // The distribution mode depends on the neighboring op's strategy
    // e.g., Conv (SOK) -> SW (SOK), the output of the Conv would be SEGMENTED
    // Conv (SOK) -> SW (Clustering), the output of the Conv would be DUPLICATED|SEGMENTED
    // The DistributedType is decided by the ops multiCluster strategy attributes
    auto greedyStrategyParentOp = getMultiClusterStrategyValue(parentOp);
    auto greedyStrategyUserOp = getMultiClusterStrategyValue(userOp);
    parentOp.setMultiClusterStrategyAttr(parentStrategy);
    userOp.setMultiClusterStrategyAttr(userStrategy);
    auto targetOutputType = getDistributedOutputType(parentOp, parentStrategy);
    auto targetInputType = getDistributedInputType(userOp, parentOp, userStrategy);
    parentOp.setMultiClusterStrategyAttr(greedyStrategyParentOp);
    userOp.setMultiClusterStrategyAttr(greedyStrategyUserOp);

    // Adjust inputType alignment for SW op
    // e.g., Conv (SOK) -> SW (SOK), the input of SW can have a same alignment with Conv
    // to avoid spilling
    auto parentOutputDistType = targetOutputType.dyn_cast<VPU::DistributedTensorType>();
    auto userInputDistType = targetInputType.dyn_cast<VPU::DistributedTensorType>();
    if (parentOutputDistType != nullptr && userInputDistType != nullptr) {
        auto parentOutAlignment = parentOutputDistType.getDistribution().alignment();
        auto UserInAlignment = userInputDistType.getDistribution().alignment();
        if (parentOutAlignment != nullptr && UserInAlignment == nullptr &&
            mlir::isa<VPU::SWOpInterface>(userOp.getOperation()) && isSWOpChannelAlignmentCompatible(userOp)) {
            targetInputType = getDistributedInputType(userOp, parentOp, userStrategy, parentOutAlignment);
        }
    }
    return {targetOutputType, targetInputType};
}

bool LayerCostModel::hasSpilling(VPU::ClusteredOpInterface clustered, VPU::ClusteredOpInterface userOp) const {
    auto targetOutputType = hasMultiClusterStrategy(clustered)
                                    ? getDistributedOutputType(clustered, getMultiClusterStrategyValue(clustered))
                                              .cast<vpux::NDTypeInterface>()
                                    : getNormalOutputType(clustered);

    auto targetInputType = hasMultiClusterStrategy(userOp)
                                   ? getDistributedInputType(userOp, clustered, getMultiClusterStrategyValue(userOp))
                                             .cast<vpux::NDTypeInterface>()
                                   : getNormalInputType(userOp, clustered);
    return hasSpilling(clustered, targetOutputType, targetInputType);
}

bool LayerCostModel::hasSpilling(VPU::ClusteredOpInterface origOp, VPU::MultiClusterStrategy origOpStrategy,
                                 VPU::ClusteredOpInterface userOp) const {
    auto targetOutputType = getDistributedOutputType(origOp, origOpStrategy).cast<vpux::NDTypeInterface>();
    auto targetInputType = hasMultiClusterStrategy(userOp)
                                   ? getDistributedInputType(userOp, origOp, getMultiClusterStrategyValue(origOp))
                                             .cast<vpux::NDTypeInterface>()
                                   : getNormalInputType(userOp, origOp);
    if (hasMultiClusterStrategy(userOp)) {
        std::tie(targetOutputType, targetInputType) =
                getDistributionTypesWithStrategy(origOp, origOpStrategy, userOp, getMultiClusterStrategyValue(userOp));
    }
    return hasSpilling(origOp, targetOutputType, targetInputType);
}

bool LayerCostModel::hasSpilling(VPU::ClusteredOpInterface origOp, VPU::ClusteredOpInterface userOp,
                                 VPU::MultiClusterStrategy userOpStrategy) const {
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    auto targetOutputType = hasMultiClusterStrategy(origOp)
                                    ? getDistributedOutputType(origOp, getMultiClusterStrategyValue(clusteredOp))
                                              .cast<vpux::NDTypeInterface>()
                                    : getNormalOutputType(origOp);
    auto targetInputType = getDistributedInputType(userOp, origOp, userOpStrategy).cast<vpux::NDTypeInterface>();
    if (hasMultiClusterStrategy(origOp)) {
        std::tie(targetOutputType, targetInputType) =
                getDistributionTypesWithStrategy(origOp, getMultiClusterStrategyValue(origOp), userOp, userOpStrategy);
    }
    return hasSpilling(origOp, targetOutputType, targetInputType);
}

bool LayerCostModel::hasSpilling(VPU::ClusteredOpInterface origOp, VPU::MultiClusterStrategy origOpStrategy,
                                 VPU::ClusteredOpInterface userOp, VPU::MultiClusterStrategy userOpStrategy) const {
    auto targetTypes = getDistributionTypesWithStrategy(origOp, origOpStrategy, userOp, userOpStrategy);
    auto targetOutputType = targetTypes.first;
    auto targetInputType = targetTypes.second;
    return hasSpilling(origOp, targetOutputType, targetInputType);
}

bool LayerCostModel::doesLayerRequireTiling(VPU::ClusteredOpInterface clusteredOp,
                                            VPU::MultiClusterStrategy strategy) const {
    auto layerStrategyChecker = LayerStrategyCheckerFactory::instance().get(clusteredOp->getName());
    return !(layerStrategyChecker->doesLayerFitIntoCMX(clusteredOp, strategy));
}

LayerCostModel::SpillingCost LayerCostModel::calculateSpillingCost(VPU::ClusteredOpInterface parentOp,
                                                                   VPU::ClusteredOpInterface userOp,
                                                                   VPU::MultiClusterStrategy parentStrategy,
                                                                   VPU::MultiClusterStrategy userStrategy) const {
    auto targetTypes = getDistributionTypesWithStrategy(parentOp, parentStrategy, userOp, userStrategy);
    auto targetOutputType = targetTypes.first;
    auto targetInputType = targetTypes.second;
    return getSpillingCost(targetOutputType, targetInputType, parentOp, userOp);
}

VPU::MultiClusterStrategy LayerCostModel::getOptimalLayerStrategy(VPU::ClusteredOpInterface clusteredOp,
                                                                  BaseLayerStrategy::Ptr layerStrategy) const {
    double splitOverHeightCost = COST_MAX;
    double splitOverKernelCost = COST_MAX;
    auto splitOverHeightFitIntoCMX = false;
    auto splitOverKernelFitIntoCMX = false;
    const auto arch = VPU::getArch(clusteredOp);
    const auto isChannelMajor = (DimsOrder::fromValue(clusteredOp->getOperand(0)) == DimsOrder::NCHW) &&
                                VPU::NCEInvariant::isChannelMajorCompatible(
                                        arch, clusteredOp->getOperand(0).getType().cast<vpux::NDTypeInterface>());
    const auto isCompressConv = VPU::NCEInvariant::isCompressConvolution(arch, clusteredOp);
    if (clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverHeight) &&
        layerStrategy->isOperationSplitOverHeightCompatible(clusteredOp)) {
        splitOverHeightCost = getLayerCost(clusteredOp, VPU::MultiClusterStrategy::SplitOverHeight);
        _log.nest().trace("SplitOverHeight cost is {0}", splitOverHeightCost);
        splitOverHeightFitIntoCMX =
                layerStrategy->doesLayerFitIntoCMX(clusteredOp, VPU::MultiClusterStrategy::SplitOverHeight);
    }

    if (clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverKernel) &&
        layerStrategy->isOperationSplitOverKernelCompatible(clusteredOp)) {
        splitOverKernelCost = getLayerCost(clusteredOp, VPU::MultiClusterStrategy::SplitOverKernel);
        _log.nest().trace("splitOverKernel cost is {0}", splitOverKernelCost);
        splitOverKernelFitIntoCMX =
                layerStrategy->doesLayerFitIntoCMX(clusteredOp, VPU::MultiClusterStrategy::SplitOverKernel);
    }

    const auto optimalHeightTiling = [&](void) {
        return (isChannelMajor || isCompressConv) ? VPU::MultiClusterStrategy::SplitOverHeightOverlapped
                                                  : VPU::MultiClusterStrategy::SplitOverHeight;
    };

    // Check if splitOverHeight is the only strategy which fits into CMX
    if (splitOverHeightFitIntoCMX && (!splitOverKernelFitIntoCMX)) {
        return optimalHeightTiling();
    }

    // Compute amount of clusters so that SOK is compatible
    const auto outputChannels =
            clusteredOp->getResult(0).getType().template cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C];
    auto uniformDistributedSegments = !VPU::isArchVPUX3XXX(VPU::getArch(clusteredOp));
    const auto sokOptimalClusters =
            getNumberOfClustersForSOKToAvoidAlignment(outputChannels, _numClusters, uniformDistributedSegments);

    // Check if splitOverKernel is the only strategy which fits into CMX and utilize full clusters
    if ((!splitOverHeightFitIntoCMX) && splitOverKernelFitIntoCMX && (sokOptimalClusters == _numClusters)) {
        return VPU::MultiClusterStrategy::SplitOverKernel;
    }

    if ((splitOverHeightCost != COST_MAX) && (splitOverKernelCost != COST_MAX) &&
        (sokOptimalClusters == _numClusters)) {
        if (!hasUserMVN(clusteredOp) && splitOverHeightCost <= splitOverKernelCost) {
            return optimalHeightTiling();
        }
        return VPU::MultiClusterStrategy::SplitOverKernel;
    }
    // SOH is P1 option as SOK may not utilize full clusters
    // However, it is still more optimal than clustering
    if (splitOverHeightCost != COST_MAX) {
        return optimalHeightTiling();
    }

    if (splitOverKernelCost != COST_MAX) {
        return VPU::MultiClusterStrategy::SplitOverKernel;
    }

    return VPU::MultiClusterStrategy::Clustering;
}

BaseLayerStrategy::Ptr LayerStrategyCheckerFactory::get(mlir::OperationName name) {
    auto clusteredOpStrategy = LayerStrategyCheckerFactory::instance()._clusteredOpStrategies.find(name);
    VPUX_THROW_WHEN(clusteredOpStrategy == LayerStrategyCheckerFactory::instance()._clusteredOpStrategies.end(),
                    "Can not find NCE op strategies for Op: {0}", name);
    return clusteredOpStrategy->second;
}

LayerStrategyCheckerFactory& LayerStrategyCheckerFactory::instance() {
    static LayerStrategyCheckerFactory obj;
    return obj;
}

void LayerStrategyCheckerFactory::registerClusteredOpStrategy(mlir::func::FuncOp func, vpux::Logger _log) {
    _clusteredOpStrategies[mlir::OperationName(NCEConvolutionOp::getOperationName(), func->getContext())] =
            std::make_shared<ConvolutionStrategy>(func, _log);
    _clusteredOpStrategies[mlir::OperationName(NCECompressConvolutionOp::getOperationName(), func->getContext())] =
            std::make_shared<CompressConvolutionStrategy>(func, _log);
    _clusteredOpStrategies[mlir::OperationName(NCEDepthConvolutionOp::getOperationName(), func->getContext())] =
            std::make_shared<DepthConvolutionStrategy>(func, _log);
    _clusteredOpStrategies[mlir::OperationName(NCEMaxPoolOp::getOperationName(), func->getContext())] =
            std::make_shared<MaxPoolStrategy>(func, _log);
    _clusteredOpStrategies[mlir::OperationName(NCEAveragePoolOp::getOperationName(), func->getContext())] =
            std::make_shared<AveragePoolStrategy>(func, _log);
    _clusteredOpStrategies[mlir::OperationName(NCEEltwiseOp::getOperationName(), func->getContext())] =
            std::make_shared<EltwiseStrategy>(func, _log);
    _clusteredOpStrategies[mlir::OperationName(NCEPermuteQuantizeOp::getOperationName(), func->getContext())] =
            std::make_shared<PermuteQuantizeStrategy>(func, _log);
    _clusteredOpStrategies[mlir::OperationName(NCEInterpolateOp::getOperationName(), func->getContext())] =
            std::make_shared<InterpolateStrategy>(func, _log);
    _clusteredOpStrategies[mlir::OperationName(VPU::TanhOp::getOperationName(), func->getContext())] =
            std::make_shared<SWGeneralStrategy>(func, _log);
    _clusteredOpStrategies[mlir::OperationName(VPU::MVNOp::getOperationName(), func->getContext())] =
            std::make_shared<SWGeneralStrategy>(func, _log);
    _clusteredOpStrategies[mlir::OperationName(VPU::SoftMaxOp::getOperationName(), func->getContext())] =
            std::make_shared<SWGeneralStrategy>(func, _log);
    _clusteredOpStrategies[mlir::OperationName(VPU::SwishOp::getOperationName(), func->getContext())] =
            std::make_shared<SWGeneralStrategy>(func, _log);
    _clusteredOpStrategies[mlir::OperationName(VPU::GeluOp::getOperationName(), func->getContext())] =
            std::make_shared<SWGeneralStrategy>(func, _log);
    _clusteredOpStrategies[mlir::OperationName(VPU::HSwishOp::getOperationName(), func->getContext())] =
            std::make_shared<SWGeneralStrategy>(func, _log);
    _clusteredOpStrategies[mlir::OperationName(VPU::InterpolateOp::getOperationName(), func->getContext())] =
            std::make_shared<SWInterpolateStrategy>(func, _log);
    _clusteredOpStrategies[mlir::OperationName(VPU::MultiplyOp::getOperationName(), func->getContext())] =
            std::make_shared<SWMultiplyStrategy>(func, _log);
    _clusteredOpStrategies[mlir::OperationName(VPU::ConcatOp::getOperationName(), func->getContext())] =
            std::make_shared<ConcatStrategy>(func, _log);
}

// For clustered op which doesn't support cycle cost calculation. The priority for straties is SOH/SOHOverlaped > SOK >
// Clustering
VPU::MultiClusterStrategy vpux::VPU::getDefaultLayerStrategy(VPU::ClusteredOpInterface clusteredOp,
                                                             BaseLayerStrategy::Ptr layerStrategy) {
    if (clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverHeight) &&
        layerStrategy->isOperationSplitOverHeightCompatible(clusteredOp)) {
        return VPU::MultiClusterStrategy::SplitOverHeight;
    }

    if (clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverHeightOverlapped) &&
        layerStrategy->isOperationSplitOverHeightCompatible(clusteredOp)) {
        return VPU::MultiClusterStrategy::SplitOverHeightOverlapped;
    }
    if (clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverKernel) &&
        layerStrategy->isOperationSplitOverKernelCompatible(clusteredOp)) {
        return VPU::MultiClusterStrategy::SplitOverKernel;
    }

    if (clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::Clustering)) {
        return VPU::MultiClusterStrategy::Clustering;
    }
    VPUX_THROW("No multi cluster strategy is supported at '{}'", clusteredOp->getLoc());
}
