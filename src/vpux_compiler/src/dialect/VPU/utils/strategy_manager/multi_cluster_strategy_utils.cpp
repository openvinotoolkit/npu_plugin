//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/strategy_manager.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include <unordered_map>

using namespace vpux;
using namespace VPU;

namespace {

double getSpillingCostForNonMultiCluster(vpux::NDTypeInterface tensorType, SpillingType /*spillingType*/,
                                         double ddrLatency, double ddrBandwidth, double /*cmxLatency*/,
                                         double /*cmxBandwidth*/, int64_t /*numDMAPorts*/) {
    // calculate the data byte size need copy from cmx to ddr or vice versa
    const auto totalSize = static_cast<double>(tensorType.getTotalAllocSize().count());
    return ddrLatency + totalSize / ddrBandwidth;
}

double getSpillingCostForDuplicated(vpux::NDTypeInterface tensorType, SpillingType /*spillingType*/, double ddrLatency,
                                    double ddrBandwidth, double /*cmxLatency*/, double /*cmxBandwidth*/,
                                    int64_t /*numDMAPorts*/) {
    auto distributedTensorType = tensorType.dyn_cast<DistributedTensorType>();
    VPUX_THROW_WHEN(distributedTensorType == nullptr, "Invalid type: {0}", tensorType);
    const auto totalSize = tensorType.getTotalAllocSize().count();
    return ddrLatency + totalSize / ddrBandwidth;
}

double getSpillingCostForSegmented(vpux::NDTypeInterface tensorType, SpillingType, double ddrLatency,
                                   double ddrBandwidth, double, double, int64_t numDMAPorts) {
    auto distributedTensorType = tensorType.dyn_cast<DistributedTensorType>();
    VPUX_THROW_WHEN(distributedTensorType == nullptr, "Invalid type: {0}", tensorType);
    auto shapes = distributedTensorType.getPerClusterComputeShapes();

    int64_t totalSize = 0;
    if (numDMAPorts > 1) {
        // For distributed segmented DMA, transaction will be split between ports and executing
        // in parallel when there are multiple DMA ports available.
        totalSize = distributedTensorType.getLargestCompactShape().totalSize();
    } else {
        // sums up of all the sub tensors to get the total size
        for (auto& shape : shapes) {
            totalSize += shape.totalSize();
        }
    }

    const Bit elemSize = tensorType.getElemTypeSize();
    totalSize = alignMemSize(elemSize * totalSize, Byte(1)).to<Byte>().count();
    return ddrLatency + static_cast<double>(totalSize) / ddrBandwidth;
}

using GetSpillingCostCB = double (*)(vpux::NDTypeInterface, SpillingType, double ddrLatency, double ddrBandwidth,
                                     double cmxLatency, double cmxBandwidth, int64_t numDMAPorts);
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

bool isSOHAlignmentCompatibleOrAdjustedCompatible(vpux::NDTypeInterface srcType, vpux::NDTypeInterface dstType) {
    const auto srcTypeInterface = srcType.dyn_cast<VPU::DistributedTypeInterface>();
    const auto dstTypeInterface = dstType.dyn_cast<VPU::DistributedTypeInterface>();
    if (srcTypeInterface == nullptr || dstTypeInterface == nullptr) {
        return false;
    }

    const auto srcDistTypes = srcTypeInterface.getDistributedTypes();
    const auto dstDistTypes = dstTypeInterface.getDistributedTypes();
    if (srcDistTypes.size() != dstDistTypes.size() || srcDistTypes.empty()) {
        return false;
    }

    const auto srcDistType = srcDistTypes.front().cast<VPU::DistributedTensorType>();
    const auto dstDistType = dstDistTypes.front().cast<VPU::DistributedTensorType>();
    if (srcDistType.getShape() != dstDistType.getShape()) {
        return false;
    }
    if (srcDistType.getDimsOrder() != dstDistType.getDimsOrder() || srcDistType.getDimsOrder() != DimsOrder::NHWC) {
        return false;
    }

    const auto srcDistAttr = srcDistType.getDistribution();
    const auto dstDistAttr = dstDistType.getDistribution();
    if (srcDistAttr.getMode().getValue() != DistributionMode::SEGMENTED ||
        dstDistAttr.getMode().getValue() != DistributionMode::SEGMENTED) {
        return false;
    }
    if (srcDistAttr.getNumTiles() != dstDistAttr.getNumTiles()) {
        return false;
    }
    if ((srcDistType.getDistribution().getMode().getValue() != DistributionMode::SEGMENTED) ||
        (dstDistType.getDistribution().getMode().getValue() != DistributionMode::SEGMENTED) ||
        (srcDistType.getDistribution().getNumTiles() != dstDistType.getDistribution().getNumTiles())) {
        return false;
    }

    return true;
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

uint32_t getLayerDMACostOverlappsWithDPU(ArrayRef<uint32_t> layerDPUCosts, ArrayRef<uint32_t> layerDMACosts,
                                         bool isDMAOverlapsWithDPU) {
    VPUX_THROW_UNLESS(layerDPUCosts.size() == layerDMACosts.size(), "Size of DPU and DMA costs should be equal.");
    VPUX_THROW_WHEN(layerDPUCosts.empty(), "DPU costs should not be empty.");

    uint32_t totalDMACost = 0;
    if (isDMAOverlapsWithDPU) {
        for (size_t tileIdx = 0; tileIdx < layerDMACosts.size() - 1; ++tileIdx) {
            if (layerDMACosts[tileIdx + 1] > layerDPUCosts[tileIdx]) {
                totalDMACost += (layerDMACosts[tileIdx + 1] - layerDPUCosts[tileIdx]);
            }
        }
    }
    totalDMACost += layerDMACosts[0];
    return totalDMACost;
}

}  // namespace

LayerCostModel::LayerCostModel(mlir::func::FuncOp func, bool enablePrefetchTiling, Logger log)
        : _func(func), _enablePrefetchTiling(enablePrefetchTiling), _log(log) {
    auto module = func->getParentOfType<mlir::ModuleOp>();

    if (auto tileOp = IE::getTileExecutor(module)) {
        auto dpuExec = tileOp.getSubExecutor(VPU::ExecutorKind::DPU);
        _NCEFrequency = tileOp.getProcessorFrequency().getValueAsDouble();
        _numTiles = tileOp.getCount();
        _numDPUs = dpuExec.getCount();
        _NCEThroughput = getNCEThroughput(VPU::getArch(tileOp));
        _DMABandwidth = getDMABandwidth(VPU::getArch(tileOp));
        if (auto shaveActExec = tileOp.getSubExecutor(ExecutorKind::SHAVE_ACT)) {
            _numShaveActs = shaveActExec.getCount();
        }
    }
    _numDMAPorts = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();
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
    if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation())) {
        auto isFilter = nceOp->getNumOperands() > 1 && input == nceOp->getOperand(1) &&
                        !mlir::isa<VPU::NCEEltwiseOp>(origOp.getOperation());
        if (isFilter) {
            return getDistributedFilterTypeFromOp(nceOp, input.getType(), numClustersAttr, specifiedStrategy);
        }
    }
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

double LayerCostModel::getDMACostOfType(vpux::NDTypeInterface srcType, SpillingType spillingType) const {
    auto distributedSrcType = srcType.dyn_cast<DistributedTensorType>();
    auto srcMode = distributedSrcType != nullptr ? distributedSrcType.getDistribution().getMode().getValue()
                                                 : VPU::DistributionMode::NONE;

    if (_arch == ArchKind::VPUX37XX) {
        return static_cast<double>(getDMACost(srcType, _vpuDeviceType, _layerCostModel, _numDMAPorts));
    }

    auto spillingReadCostFunc = spillingCostMap.at(srcMode);
    return spillingReadCostFunc(srcType, spillingType, _DDRLatency, _DMABandwidth, _CMXLatency, _CMXMulticastBandwidth,
                                _numDMAPorts);
}

double LayerCostModel::getSpillingDMACost(vpux::NDTypeInterface srcTensorType, SpillingType spillingType) const {
    if (auto sparseTensorType = srcTensorType.dyn_cast<VPU::SparseTensorType>()) {
        srcTensorType = sparseTensorType.getData().cast<vpux::NDTypeInterface>();
    }
    return getDMACostOfType(srcTensorType, spillingType);
}

double LayerCostModel::getSpillingReadCost(vpux::NDTypeInterface srcTensorType) const {
    return getSpillingDMACost(srcTensorType, SpillingType::SPILL_READ);
}

double LayerCostModel::getSpillingWriteCost(vpux::NDTypeInterface srcTensorType) const {
    return getSpillingDMACost(srcTensorType, SpillingType::SPILL_WRITE);
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

    double clusterOpsPerCycle = _NCEThroughput / _NCEFrequency / _numTiles;
    double clusterEff = computeSplitEfficiency(nceOp, strategy);
    auto largestClusterOutShape = getLargestClusterOutputShape(clusteredOp, strategy);

    auto kernelSize = nceOp.getKernelSizeVal();
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
    const int64_t IC = (mlir::isa<VPU::NCEConvolutionOp, VPU::NCECompressConvolutionOp, VPU::NCEInterpolateOp>(op))
                               ? getShape(op->getOperand(0))[Dims4D::Act::C]
                               : 1;
    const int64_t OC = getShape(op->getResult(0))[Dims4D::Act::C];
    const auto kernelSize = nceOp.getKernelSizeVal();
    auto numClustersAttr = VPU::getOptimalNumClusters(clusteredOp, OC, strategy);

    /// Weights cost
    /// Weights and weightTable are Segmented mode under SOK (only including ddr -> cmx cost),
    /// SOK may use less clusters to avoid alignment
    /// So it's not proper to estimate total weightsSize by "clusterWeightsSize * _numTiles" simply
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

                const Bit elemBitSize = getElemTypeSize(distributedWeightsType);
                int64_t weightsByteSize = 0;

                if (type.index() == 0 && !numElemsPerOC.empty()) {
                    for (auto p : zip(tiledWeightsShapes, tiledWeightsOffsets)) {
                        const auto tileShape = std::get<0>(p);
                        const auto tileOffsets = std::get<1>(p);
                        const auto startOC = tileOffsets[Dims4D::Filter::OC];
                        const auto endOC = startOC + tileShape[Dims4D::Filter::OC];
                        for (auto idx = startOC; idx < endOC; ++idx) {
                            const auto numElems = *(numElemsPerOC.begin() + idx);
                            weightsByteSize +=
                                    alignMemSize(elemBitSize * numElems, Byte(weightSetAlignment)).to<Byte>().count();
                        }
                    }
                } else {
                    for (auto& clusterWeightsShape : tiledWeightsShapes) {
                        weightsByteSize +=
                                alignMemSize(elemBitSize * clusterWeightsShape.totalSize(), Byte(1)).to<Byte>().count();
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
            const Bit elemBiSize = ndWeightsType.getElemTypeSize();
            const int64_t weightSetSize =
                    IC * kernelSize[Dims4D::Kernel::Y.ind()] * kernelSize[Dims4D::Kernel::X.ind()];

            int64_t clusterWeightsSize = 0;
            if (!numElemsPerOC.empty()) {
                for (auto numElems : numElemsPerOC) {
                    clusterWeightsSize +=
                            alignMemSize(elemBiSize * numElems, Byte(_cmxAddressAlignment)).to<Byte>().count();
                }
            } else {
                clusterWeightsSize +=
                        OC * alignMemSize(elemBiSize * weightSetSize, Byte(_cmxAddressAlignment)).to<Byte>().count();
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
    if (nceOp.getWeightsTableOperand() != nullptr) {
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
    if (nceOp.getActivationWindowOperand() != nullptr) {
        const auto SX = Shape(nceOp.getStridesVal())[Dims4D::Strides::X];
        const auto inputElemType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>().getElementType();
        auto sparsityMode = VPU::NCESparsity::Mode::POOL;
        if (mlir::isa<VPU::NCEConvolutionOp>(op)) {
            sparsityMode = VPU::NCESparsity::Mode::CM_CONV;
        } else if (mlir::isa<VPU::NCEDepthConvolutionOp>(op)) {
            sparsityMode = VPU::NCESparsity::Mode::DW_CONV;
        }

        const auto activationWindowSize =
                VPU::NCESparsity::getActivationWindowSize(sparsityMode, Shape(kernelSize), SX, inputElemType, IC);
        totalActivationWindowCycles = _DDRLatency + (static_cast<double>(activationWindowSize) / _DMABandwidth);
    }

    /// PWLTable cost
    /// DistributionMode is Duplicated, at most we have 32 instructions (8 segments for pwl),
    /// so size is 32 x INT32 = 128 Bytes
    if (auto origOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(op)) {
        if (origOp.getInstructionListTable()) {
            pwlTableCycles = _DDRLatency + (128.0 / _DMABandwidth);
        }
    } else if (auto origOp = mlir::dyn_cast<VPU::NCEDepthConvolutionOp>(op)) {
        if (origOp.getInstructionListTable()) {
            pwlTableCycles = _DDRLatency + (128.0 / _DMABandwidth);
        }
    }

    // Total cost for single layer
    return totalActivationWindowCycles + totalWeightCycles + totalWeightsTableCycles + pwlTableCycles + outputCycles;
}

/// @brief A switcher to select time-cost or efficiency-cost for greedy strategy assignment
/// @details Time-cost includes an extra input spilling cost to be more accurate
double LayerCostModel::getNCELayerCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy,
                                       bool useTimeBasedCost) {
    if (!useTimeBasedCost) {
        return getEfficiencyCost(nceOp, strategy);
    }

    double basicDPUandDMACost = COST_MAX;

    const auto it = _costCache.find(nceOp);
    if (it == _costCache.end()) {
        // Case 1 - Op costs are not found in cache:
        // 1.Calculate cost value with VPUNN
        // 2.Create new op costs
        // 3.Store the new op costs in cost cache
        basicDPUandDMACost = getDPUandDMATimeCost(nceOp, strategy);
        SmallVector newOpCosts((getMaxEnumValForMultiClusterStrategy() + 1), COST_MAX);
        newOpCosts[static_cast<uint64_t>(strategy)] = basicDPUandDMACost;
        _costCache.insert({nceOp, newOpCosts});
    } else {
        auto strategyCostIt = it->second.begin() + static_cast<uint64_t>(strategy);
        if (strategyCostIt != nullptr && *strategyCostIt != COST_MAX) {
            // Case 2 - Strategy cost is found in cache:
            // Retrieve the cost value directly
            basicDPUandDMACost = *strategyCostIt;
        } else {
            // Case 3 - Op costs are found but op strategy cost is not found:
            // 1.Calculate cost value with VPUNN
            // 2.Update op strategy cost value in cache
            basicDPUandDMACost = getDPUandDMATimeCost(nceOp, strategy);
            *strategyCostIt = basicDPUandDMACost;
        }
    }

    return basicDPUandDMACost;
}

/// @brief Time-cost : return the shave computation time (cycles)
/// @details use vpunn cost model to get the shave cost of sw layer
double LayerCostModel::getSWLayerCost(VPU::SWOpInterface swOp, VPU::MultiClusterStrategy strategy) const {
    auto getVPUTensors = [&](mlir::ValueRange values) -> std::vector<VPUNN::VPUTensor> {
        std::vector<VPUNN::VPUTensor> tensors;
        std::transform(values.begin(), values.end(), std::back_inserter(tensors), [](mlir::Value value) {
            auto valueType = value.getType().cast<vpux::NDTypeInterface>();
            return VPU::getVPUTensor(valueType.getShape(), valueType.getElementType());
        });
        return tensors;
    };

    const auto device = VPU::getVPUDeviceType(_arch);
    const auto inputTensors = getVPUTensors(swOp->getOperands());
    const auto outputTensors = getVPUTensors(swOp->getResults());

    std::shared_ptr<VPUNN::SWOperation> vpunnLayer;
    llvm::TypeSwitch<mlir::Operation*, void>(swOp.getOperation())
            .Case<VPU::TanhOp>([&](VPU::TanhOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVTanh>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::MVNOp>([&](VPU::MVNOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVMVN>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::SoftMaxOp>([&](VPU::SoftMaxOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVSoftmax>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::SwishOp>([&](VPU::SwishOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVSwish>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::HSwishOp>([&](VPU::HSwishOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVHardSwish>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::PowerOp>([&](VPU::PowerOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVPower>(device, inputTensors, outputTensors.front());
            })
            .Case<VPU::DivideOp>([&](VPU::DivideOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVDivide>(device, inputTensors, outputTensors.front());
            })
            .Default([&](mlir::Operation* op) {
                VPUX_THROW("SW op {0} has no VPUNN support", op->getName());
            });
    auto vpunnStrategy = VPU::getVPULayerStrategy(strategy, _numDPUs, _numTiles, _numShaveActs, false);
    return _layerCostModel->Layer(*vpunnLayer, vpunnStrategy);
}

/// @brief get computation cost
/// @details Time-cost includes an extra input spilling cost to be more accurate
double LayerCostModel::getLayerCost(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy,
                                    bool useTimeBasedCost) {
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

double LayerCostModel::getDPUandDMATimeCostWithCustomTiling(VPU::NCEOpInterface nceOp,
                                                            VPU::MultiClusterStrategy strategy,
                                                            const OutputTiling& outTiles) const {
    // Types for each tile
    SmallVector<SmallVector<NDTypeInterface>> tilesTypes;

    _log.trace("Start calculating VPUNN layer cost for Op {0} with strategy {1}", nceOp.getLoc(), strategy);

    const auto costParams = VPU::getWorkloadCostParam(nceOp, _arch, _numDPUs);
    const auto vpunnStrategy = VPU::getVPULayerStrategy(strategy, _numDPUs, _numTiles, 1, true);
    auto vpunnLayerDPUCosts =
            getDPUCostForNCEOp(nceOp, strategy, outTiles, tilesTypes, costParams, vpunnStrategy, _layerCostModel, _log);
    if (vpunnLayerDPUCosts.empty()) {
        return COST_MAX;
    }
    _log.trace("VPUNN DPU layer costs {0}", vpunnLayerDPUCosts);

    const auto getSpillingReadCost = [&](NDTypeInterface srcType) -> uint32_t {
        return checked_cast<uint32_t>(this->getSpillingReadCost(srcType));
    };

    double cost = 0;

    // Accumulate all the DPU costs
    cost += std::accumulate(vpunnLayerDPUCosts.begin(), vpunnLayerDPUCosts.end(), 0);

    // Add weights DMA costs
    auto vpunnLayerWeightsCosts = getPerTileWeightsDMACosts(nceOp, tilesTypes, getSpillingReadCost);
    _log.trace("VPUNN weights DMA costs {0}", vpunnLayerWeightsCosts);
    cost += getWeightsDMACostForNCEOp(nceOp, outTiles, vpunnLayerDPUCosts, vpunnLayerWeightsCosts,
                                      _enablePrefetchTiling, _log);

    // Add activation DMA costs
    auto vpunnLayerActCosts = getPerTileActivationDMACosts(nceOp, tilesTypes, getSpillingReadCost);
    _log.trace("VPUNN activation DMA costs {0}", vpunnLayerActCosts);
    cost += getActivationDMACostForNCEOp(nceOp, outTiles, vpunnLayerDPUCosts, vpunnLayerActCosts, _enablePrefetchTiling,
                                         _log);

    // Add output spilling cost
    // for non clusteredOp, must be ops that requires tiling
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    if (clusteredOp == nullptr || !clusteredOp.doesLayerFitIntoCMX(strategy, /*reservedMem=*/Byte(0))) {
        auto outputType = getDistributedOutputType(clusteredOp, strategy);
        auto outputSpillingCost = getSpillingWriteCost(outputType);
        _log.trace("VPUNN output spilling cost {0}", outputSpillingCost);
        cost += outputSpillingCost;
    }

    return cost;
}

/// @brief Time-cost : return a sum of layer DPU time and weights DMA time (cycles)
/// @details DPU time calculation also considers the impact of workloads split efficiency
double LayerCostModel::getDPUandDMATimeCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    if (_arch == ArchKind::VPUX37XX) {
        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
        VPUX_THROW_WHEN(clusteredOp == nullptr, "NCE op {0} at {1} should be a clustered op", nceOp->getName(),
                        nceOp.getLoc());

        // Set customized strategy to the op to get corresponding output tiles when tiling
        // Save and restore original strategy if needed
        auto origStrategy = clusteredOp.getMultiClusterStrategy();
        clusteredOp.setMultiClusterStrategy(strategy);

        // Output tiling for each tile
        OutputTiling outTiles({TileInfo(getShape(nceOp->getResult(0)))});

        // Check CMX memory as VPUNN works with layer which fits CMX memory
        // If not, tiling big layer to fit into CMX
        if (!(clusteredOp.doesLayerFitIntoCMX(strategy, /*reservedMem=*/Byte(0)))) {
            _log.trace("Tiling op {0} to fit into cmx before passing to VPUNN Layer API", nceOp.getLoc());
            auto tilingBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(nceOp.getOperation());
            VPUX_THROW_WHEN(tilingBuilderOp == nullptr, "NCE op {0} at {1} should be a tiling op", nceOp->getName(),
                            nceOp.getLoc());

            auto tiles = getLayerTilingStrategy(tilingBuilderOp, _enablePrefetchTiling, _log);
            if (mlir::failed(tiles)) {
                _log.trace("Invalid tiling strategy for {0}", nceOp->getName());
                return COST_MAX;
            }
            outTiles = tiles.value();
        }

        auto cost = getDPUandDMATimeCostWithCustomTiling(nceOp, strategy, outTiles);

        _log.trace("VPUNN total layer cost for {0} is {1}", strategy, cost);

        // Restore original strategy or remove temporary strategy
        if (origStrategy.has_value()) {
            clusteredOp.setMultiClusterStrategy(origStrategy.value());
        } else {
            clusteredOp->removeAttr(multiClusterStrategy);
        }

        return cost;
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
        return clusteringOp.getMultiClusterStrategy().has_value();
    }

    return false;
}

VPU::MultiClusterStrategy LayerCostModel::getMultiClusterStrategyValue(VPU::ClusteredOpInterface clusteredOp) const {
    auto strategy = clusteredOp.getMultiClusterStrategy();
    if (!strategy.has_value()) {
        VPUX_THROW("NCE operation {0} doesn't have multiClusterStrategy attribute", clusteredOp->getLoc());
    }

    return strategy.value();
}

/// Anywhere if you need to judge spilling existing, please call me!
/// srcTensorType is the output of parent origOp
/// dstTensorType is the input of child NCE op
bool LayerCostModel::hasSpilling(VPU::ClusteredOpInterface /*clusteredOp*/, vpux::NDTypeInterface srcTensorType,
                                 vpux::NDTypeInterface dstTensorType) const {
    if (isTargetTensorTypeCompatible(srcTensorType, dstTensorType) ||
        isSOHAlignmentCompatibleOrAdjustedCompatible(srcTensorType, dstTensorType)) {
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
    parentOp.setMultiClusterStrategy(parentStrategy);
    userOp.setMultiClusterStrategy(userStrategy);
    auto targetOutputType = getDistributedOutputType(parentOp, parentStrategy);
    auto targetInputType = getDistributedInputType(userOp, parentOp, userStrategy);
    parentOp.setMultiClusterStrategy(greedyStrategyParentOp);
    userOp.setMultiClusterStrategy(greedyStrategyUserOp);

    // Adjust inputType alignment for SW op
    // e.g., Conv (SOK) -> SW (SOK), the input of SW can have a same alignment with Conv
    // to avoid spilling
    auto parentOutputDistType = targetOutputType.dyn_cast<VPU::DistributedTensorType>();
    auto userInputDistType = targetInputType.dyn_cast<VPU::DistributedTensorType>();
    if (parentOutputDistType != nullptr && userInputDistType != nullptr) {
        auto parentOutAlignment = parentOutputDistType.getDistribution().getAlignment();
        auto UserInAlignment = userInputDistType.getDistribution().getAlignment();
        if (parentOutAlignment != nullptr && UserInAlignment == nullptr &&
            mlir::isa<VPU::SWOpInterface>(userOp.getOperation()) &&
            isSWOpChannelAlignmentCompatible(userOp, targetInputType,
                                             userOp->getResult(0).getType().cast<vpux::NDTypeInterface>())) {
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
    return !(clusteredOp.doesLayerFitIntoCMX(strategy, /*reservedMem=*/Byte(0)));
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

VPU::MultiClusterStrategy LayerCostModel::getOptimalLayerStrategy(VPU::ClusteredOpInterface clusteredOp) {
    double splitOverHeightCost = COST_MAX;
    double splitOverKernelCost = COST_MAX;
    auto splitOverHeightFitIntoCMX = false;
    auto splitOverKernelFitIntoCMX = false;
    const auto arch = VPU::getArch(clusteredOp);
    const auto isChannelMajor = (DimsOrder::fromValue(clusteredOp->getOperand(0)) == DimsOrder::NCHW) &&
                                VPU::NCEInvariant::isChannelMajorCompatible(
                                        arch, clusteredOp->getOperand(0).getType().cast<vpux::NDTypeInterface>());
    if (clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverHeight) &&
        clusteredOp.isOperationSplitOverHeightCompatible(/*vpux::TileInfo=*/vpux::TileInfo(ShapeRef()))) {
        splitOverHeightCost = getLayerCost(clusteredOp, VPU::MultiClusterStrategy::SplitOverHeight);
        _log.nest().trace("SplitOverHeight cost is {0}", splitOverHeightCost);
        splitOverHeightFitIntoCMX =
                clusteredOp.doesLayerFitIntoCMX(VPU::MultiClusterStrategy::SplitOverHeight, /*reservedMem=*/Byte(0));
    }

    if (clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverKernel) &&
        clusteredOp.isOperationSplitOverKernelCompatible(/*outputShape=*/ShapeRef(), /*offset=*/ShapeRef(),
                                                         /*axis=*/ShapeRef())) {
        splitOverKernelCost = getLayerCost(clusteredOp, VPU::MultiClusterStrategy::SplitOverKernel);
        _log.nest().trace("SplitOverKernel cost is {0}", splitOverKernelCost);
        splitOverKernelFitIntoCMX =
                clusteredOp.doesLayerFitIntoCMX(VPU::MultiClusterStrategy::SplitOverKernel, /*reservedMem=*/Byte(0));
    }

    const auto optimalHeightTiling = [&](void) {
        return (isChannelMajor || mlir::isa<vpux::VPU::NCECompressConvolutionOp, vpux::VPU::NCEPermuteOp>(clusteredOp))
                       ? VPU::MultiClusterStrategy::SplitOverHeightOverlapped
                       : VPU::MultiClusterStrategy::SplitOverHeight;
    };

    // Check if SplitOverHeight is the only strategy which fits into CMX
    if (splitOverHeightFitIntoCMX && (!splitOverKernelFitIntoCMX)) {
        return optimalHeightTiling();
    }

    // Compute amount of clusters so that SOK is compatible
    const auto outputChannels =
            clusteredOp->getResult(0).getType().template cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C];
    auto uniformDistributedSegments = !VPU::isArchVPUX3XXX(VPU::getArch(clusteredOp));
    const auto sokOptimalClusters =
            getNumberOfClustersForSOKToAvoidAlignment(outputChannels, _numTiles, uniformDistributedSegments);

    // Check if SplitOverKernel is the only strategy which fits into CMX and utilize full clusters
    if ((!splitOverHeightFitIntoCMX) && splitOverKernelFitIntoCMX && (sokOptimalClusters == _numTiles)) {
        return VPU::MultiClusterStrategy::SplitOverKernel;
    }

    if ((splitOverHeightCost != COST_MAX) && (splitOverKernelCost != COST_MAX) && (sokOptimalClusters == _numTiles)) {
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

// For clustered op which doesn't support cycle cost calculation. The priority for straties is SOH/SOHOverlaped > SOK >
// Clustering
VPU::MultiClusterStrategy vpux::VPU::getDefaultLayerStrategy(VPU::ClusteredOpInterface clusteredOp) {
    if (clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverHeight) &&
        clusteredOp.isOperationSplitOverHeightCompatible(/*vpux::TileInfo=*/vpux::TileInfo(ShapeRef()))) {
        return VPU::MultiClusterStrategy::SplitOverHeight;
    }

    if (clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverHeightOverlapped) &&
        clusteredOp.isOperationSplitOverHeightCompatible(/*vpux::TileInfo=*/vpux::TileInfo(ShapeRef()))) {
        return VPU::MultiClusterStrategy::SplitOverHeightOverlapped;
    }
    if (clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverKernel) &&
        clusteredOp.isOperationSplitOverKernelCompatible(/*outputShape=*/ShapeRef(), /*offset=*/ShapeRef(),
                                                         /*axis=*/ShapeRef())) {
        return VPU::MultiClusterStrategy::SplitOverKernel;
    }

    if (mlir::isa<VPU::SoftMaxOp, VPU::DepthToSpaceOp>(clusteredOp.getOperation()) &&
        clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverWidth) &&
        clusteredOp.isOperationSplitOverWidthCompatible(/*outputShape=*/ShapeRef(), /*offset=*/ShapeRef(),
                                                        /*axis=*/ShapeRef())) {
        return VPU::MultiClusterStrategy::SplitOverWidth;
    }

    if (clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::Clustering)) {
        return VPU::MultiClusterStrategy::Clustering;
    }
    VPUX_THROW("No multi cluster strategy is supported at '{}'", clusteredOp->getLoc());
}

bool vpux::VPU::isStrategyCompatibleShape(VPU::ClusteredOpInterface clusteredOp, const vpux::TileInfo& outputTile,
                                          VPU::MultiClusterStrategy strategy, Logger log) {
    auto shape = ShapeRef(outputTile.shape);
    if (shape.size() != RANK_REQUIRED_FOR_TILING) {
        log.trace("Operation '{0}' at '{1}' has output rank {2} and cannot be tiled. Expected rank: {3}.",
                  clusteredOp->getName(), clusteredOp->getLoc(), shape.size(), RANK_REQUIRED_FOR_TILING);
        return false;
    }
    switch (strategy) {
    case MultiClusterStrategy::SplitOverHeight:
    case MultiClusterStrategy::SplitOverHeightOverlapped:
    case MultiClusterStrategy::HKSwitch: {
        return clusteredOp.isOperationSplitOverHeightCompatible(outputTile);
    }
    case MultiClusterStrategy::SplitOverHeightKernel: {
        return clusteredOp.isOperationSplitOverHeightCompatible(outputTile) &&
               clusteredOp.isOperationSplitOverKernelCompatible(outputTile.shape, outputTile.offsets, outputTile.axis);
    }
    case MultiClusterStrategy::SplitOverWidth: {
        return clusteredOp.isOperationSplitOverWidthCompatible(outputTile.shape, outputTile.offsets, outputTile.axis);
    }
    case MultiClusterStrategy::SplitOverKernel: {
        return clusteredOp.isOperationSplitOverKernelCompatible(outputTile.shape, outputTile.offsets, outputTile.axis);
    }
    case MultiClusterStrategy::SplitOverHeightWidth: {
        return clusteredOp.isOperationSplitOverHeightCompatible(outputTile) &&
               clusteredOp.isOperationSplitOverWidthCompatible(outputTile.shape, outputTile.offsets, outputTile.axis);
    }
    case MultiClusterStrategy::Clustering: {
        return true;
    }
    default: {
        VPUX_THROW("Unknown multi cluster strategy {0}", strategy);
    }
    }
}

SmallVector<uint32_t> vpux::VPU::getDPUCostForNCEOp(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy mcStrategy,
                                                    const OutputTiling& outTiles,
                                                    SmallVector<SmallVector<NDTypeInterface>>& tilesTypes,
                                                    const VPUIP::WorkloadCostParams& costParams,
                                                    VPUNN::VPULayerStrategy vpunnStrategy,
                                                    const std::shared_ptr<VPUNN::VPULayerCostModel>& vpunnCostModel,
                                                    Logger log) {
    SmallVector<VPUNN::DPULayer> vpunnLayers{VPU::getDPULayer(costParams)};
    if (!outTiles.empty()) {
        auto tilingBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(nceOp.getOperation());
        VPUX_THROW_WHEN(tilingBuilderOp == nullptr, "NCE op {0} at {1} should be a tiling op", nceOp->getName(),
                        nceOp.getLoc());
        const auto tilingVPUNNLayer = [&](const VPUNN::DPULayer& vpunnLayer,
                                          const OutputTiling& outTiles) -> SmallVector<VPUNN::DPULayer> {
            SmallVector<VPUNN::DPULayer> vpunnLayers;
            vpunnLayers.reserve(outTiles.size());
            for (auto& outTile : outTiles) {
                vpunnLayers.push_back(vpunnLayer);
                auto inTiles = tilingBuilderOp.backInferTileInfo(outTile, log);
                tilesTypes.push_back(getTileTypes(nceOp.getOperation(), outTile, inTiles));
                auto& inputTile = inTiles.tiles.front();
                auto inPad = inTiles.pads;
                vpunnLayers.back().inputs = {getVPUTensor(inputTile.shape, costParams.inDataType)};
                vpunnLayers.back().outputs = {getVPUTensor(outTile.shape, costParams.outDataType)};
                if (inPad.has_value()) {
                    vpunnLayers.back().padding = {
                            static_cast<unsigned int>(inPad->top), static_cast<unsigned int>(inPad->bottom),
                            static_cast<unsigned int>(inPad->left), static_cast<unsigned int>(inPad->right)};
                }
            }
            return vpunnLayers;
        };
        vpunnLayers = tilingVPUNNLayer(vpunnLayers[0], outTiles);
    }

    SmallVector<uint32_t> layerDPUCosts;
    for (auto& vpunnLayer : vpunnLayers) {
        auto cost = checkAndReturnCost(vpunnCostModel->Layer(vpunnLayer, vpunnStrategy), log);
        if (cost >= VPU::INVALID_COST_BASE) {
            printVPUNNLayerConfig(vpunnLayer, vpunnStrategy, log);
            if (cost == VPU::ERROR_INPUT_TOO_BIG && !layerDPUCosts.empty()) {
                log.trace(" Use the first availabe layer cost to estimate the layer with ERROR_INPUT_TOO_BIG");
                cost = layerDPUCosts.front();
            } else {
                layerDPUCosts.clear();
                break;
            }
        }
        if (mlir::isa<VPU::NCEEltwiseOp>(nceOp.getOperation()) &&
            (mcStrategy == VPU::MultiClusterStrategy::Clustering)) {
            // The VPUNN cost of NCEEltwiseOp is inaccurate
            // Multiply a ratio to correct the cost
            // Track [E#98656]
            cost *= NCEELTWISE_DPU_COST_RATIO;
        }
        layerDPUCosts.push_back(cost);
    }

    return layerDPUCosts;
}

SmallVector<uint32_t> vpux::VPU::getPerTileWeightsDMACosts(
        VPU::NCEOpInterface nceOp, ArrayRef<SmallVector<NDTypeInterface>> tilesTypes,
        std::function<uint32_t(NDTypeInterface)> getSpillingReadCostFunc) {
    auto weightsOperand = nceOp.getWeightsOperand();
    if (weightsOperand == nullptr) {
        return SmallVector<uint32_t>(std::max<size_t>(tilesTypes.size(), 1), 0);
    }

    const auto inferredTileTypes = SmallVector<SmallVector<NDTypeInterface>>{
            getTileTypes(nceOp.getOperation(), TileInfo(getShape(nceOp->getResult(0))))};
    const auto& typesList = tilesTypes.empty() ? inferredTileTypes : tilesTypes;

    SmallVector<uint32_t> perTileWeightsCosts;
    for (const auto& tileTypes : typesList) {
        VPUX_THROW_UNLESS(tileTypes.size() > 1,
                          "NCEOp {0} at {1} has invalid number of tile types, got {2}, expected >1", nceOp->getName(),
                          nceOp->getLoc(), tileTypes.size());
        auto weightsDMACost = checked_cast<uint32_t>(getSpillingReadCostFunc(tileTypes[1]));
        perTileWeightsCosts.push_back(weightsDMACost);
    }

    return perTileWeightsCosts;
}

SmallVector<uint32_t> vpux::VPU::getPerTileActivationDMACosts(
        VPU::NCEOpInterface nceOp, ArrayRef<SmallVector<NDTypeInterface>> tilesTypes,
        std::function<uint32_t(NDTypeInterface)> getSpillingReadCostFunc) {
    auto getParentOp = [&]() {
        mlir::Operation* parentOp = nceOp->getOperand(0).getDefiningOp();
        while (parentOp && (mlir::isa<VPU::GroupSparseTensorOp>(parentOp) || isPureViewOp(parentOp))) {
            parentOp = parentOp->getOperand(0).getDefiningOp();
        }

        return parentOp;
    };

    // If op fit into CMX and parent op exists, we assume act spilling can be removed by adjusting startegy
    if ((tilesTypes.size() <= 1) && (getParentOp() != nullptr)) {
        return SmallVector<uint32_t>(std::max<size_t>(tilesTypes.size(), 1), 0);
    }

    bool isEltwiseOpWithDiffInputs =
            (mlir::isa<VPU::NCEEltwiseOp>(nceOp) && nceOp->getOperand(0) != nceOp->getOperand(1));

    SmallVector<uint32_t> perTileActCosts;
    for (const auto& tileTypes : tilesTypes) {
        VPUX_THROW_UNLESS(tileTypes.size() > 1,
                          "NCEOp {0} at {1} has invalid number of tile types, got {2}, expected >1", nceOp->getName(),
                          nceOp->getLoc(), tileTypes.size());
        auto actDMACost = checked_cast<uint32_t>(getSpillingReadCostFunc(tileTypes[0]));
        if (isEltwiseOpWithDiffInputs) {
            actDMACost += checked_cast<uint32_t>(getSpillingReadCostFunc(tileTypes[1]));
        }
        perTileActCosts.push_back(actDMACost);
    }

    return perTileActCosts;
}

uint32_t vpux::VPU::getWeightsDMACostForNCEOp(VPU::NCEOpInterface nceOp, const OutputTiling& outTiles,
                                              ArrayRef<uint32_t> layerDPUCosts, ArrayRef<uint32_t> layerDMACosts,
                                              bool enablePrefetchTiling, vpux::Logger log) {
    VPUX_THROW_WHEN(layerDPUCosts.empty() || layerDPUCosts.size() != layerDMACosts.size(),
                    "Layer DPU costs must be non-empty and equal to DMA costs in size");

    const auto outShape = getShape(nceOp->getResult(0));
    auto tiles = outTiles.empty() ? OutputTiling({TileInfo(outShape)}) : outTiles;

    auto weightsOperand = nceOp.getWeightsOperand();
    bool isWeightsDMASplitOnEachTile = (weightsOperand != nullptr && tiles.front().axis[Dims4D::Act::C] > 1);

    auto tilingInfoOp = mlir::dyn_cast<VPU::TilingInfoOpInterface>(nceOp.getOperation());
    // The first weights DMA should not be counted, overlapping with the parent op
    bool isFirstWeightsDMAOverlappedWithParent =
            enablePrefetchTiling ? tilingInfoOp != nullptr &&
                                           tilingInfoOp.isSupportedTiling(tiles, vpux::TilingMode::PREFETCHING, log)
                                 : false;
    // If the DMA will overlap with DPU from the second tile on
    bool isDMAOverlappedWithDPU =
            enablePrefetchTiling ? tilingInfoOp != nullptr &&
                                           tilingInfoOp.isSupportedTiling(tiles, vpux::TilingMode::PIPELINING, log)
                                 : false;

    uint32_t totalDMACost = 0;

    if (isDMAOverlappedWithDPU) {
        // Weights DMA from second tile on will be overlapped with DPU of previous tile
        totalDMACost += getLayerDMACostOverlappsWithDPU(layerDPUCosts, layerDMACosts, isWeightsDMASplitOnEachTile);
    } else {
        // When DMA not overlapped with DPU
        //  - If weights DMA will be copied on each tile, we need to accumulate all the DMA costs
        //  - If weights DMA will be shared for all tiles, we only add the first DMA cost
        totalDMACost += isWeightsDMASplitOnEachTile ? std::accumulate(layerDMACosts.begin(), layerDMACosts.end(), 0U)
                                                    : layerDMACosts.front();
    }

    if (isFirstWeightsDMAOverlappedWithParent) {
        // the first DMA will overlap with previous op's DPU, so exclude it from cost
        totalDMACost -= layerDMACosts.front();
    }

    return totalDMACost;
}

uint32_t vpux::VPU::getActivationDMACostForNCEOp(VPU::NCEOpInterface nceOp, const OutputTiling& outTiles,
                                                 ArrayRef<uint32_t> layerDPUCosts, ArrayRef<uint32_t> layerDMACosts,
                                                 bool enablePrefetchTiling, vpux::Logger log) {
    VPUX_THROW_WHEN(layerDPUCosts.empty() || layerDPUCosts.size() != layerDMACosts.size(),
                    "Layer DPU costs must be non-empty and equal to DMA costs in size");

    const auto outShape = getShape(nceOp->getResult(0));
    auto tiles = outTiles.empty() ? OutputTiling({TileInfo(outShape)}) : outTiles;

    auto weightsOperand = nceOp.getWeightsOperand();
    bool isActDMASplitOnEachTile = (weightsOperand == nullptr || tiles.front().axis[Dims4D::Act::C] == 1);

    auto tilingInfoOp = mlir::dyn_cast<VPU::TilingInfoOpInterface>(nceOp.getOperation());
    // The DMA will overlap with DPU from the second tile on
    bool isDMAOverlappedWithDPU =
            enablePrefetchTiling ? tilingInfoOp != nullptr &&
                                           tilingInfoOp.isSupportedTiling(tiles, vpux::TilingMode::PIPELINING, log)
                                 : false;

    uint32_t totalDMACost = 0;

    if (isDMAOverlappedWithDPU) {
        // Act DMA from second tile on will be overlapped with DPU of previous tile
        totalDMACost += getLayerDMACostOverlappsWithDPU(layerDPUCosts, layerDMACosts, isActDMASplitOnEachTile);
    } else {
        // When DMA not overlapped with DPU
        //  - If act DMA will be copied on each tile, we need to accumulate all the DMA costs
        //  - If act DMA will be shared for all tiles, we only add the first DMA cost
        totalDMACost += isActDMASplitOnEachTile ? std::accumulate(layerDMACosts.begin(), layerDMACosts.end(), 0U)
                                                : layerDMACosts.front();
    }

    return totalDMACost;
}

size_t vpux::VPU::getNumNonConstantOperands(mlir::Operation* op) {
    return std::count_if(op->operand_begin(), op->operand_end(), [](mlir::Value operand) {
        return !mlir::isa_and_nonnull<Const::DeclareOp>(operand.getDefiningOp());
    });
}

bool vpux::VPU::hasLayerWithMultipleInputs(mlir::Operation* op) {
    return std::any_of(op->user_begin(), op->user_end(), [](mlir::Operation* user) {
        return getNumNonConstantOperands(user) > 1 || hasLayerWithMultipleInputs(user);
    });
}
