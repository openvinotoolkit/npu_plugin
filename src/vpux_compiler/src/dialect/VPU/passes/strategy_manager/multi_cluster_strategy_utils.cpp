//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/strategy_manager.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include <unordered_map>

using namespace vpux;
using namespace VPU;

namespace {

enum class SpillingType { SPILL_WRITE, SPILL_READ };

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

mlir::Value getInputFromClusteredOp(VPU::ClusteredOpInterface nceOp, mlir::Operation* parentOp) {
    for (auto operand : nceOp->getOperands()) {
        auto parent = operand.getDefiningOp();
        if (parent == parentOp) {
            return operand;
        }
        while (mlir::isa<VPU::ShapeCastOp, VPU::QuantizeCastOp>(parent)) {
            // propagate cast ops
            parent = parent->getOperand(0).getDefiningOp();
            if (parent == parentOp) {
                return operand;
            }
        }
    }
    VPUX_THROW("Cannot find input from op: {0}, parent op: {1}", nceOp, *parentOp);
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

bool isAlignmentCompatible(VPU::ClusteredOpInterface nceOp, vpux::NDTypeInterface srcType,
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
        if (distSrcDataType.getDistribution().mode().getValue() == DistributionMode::SEGMENTED) {
            const auto newDistributedOutputType = adjustOutputAlignmentForSOH(nceOp, dstType);
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

LayerCostModel::LayerCostModel(mlir::FuncOp func, Logger log): _func(func), _log(log) {
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto nceOp = IE::getAvailableExecutor(module, ExecutorKind::NCE);
    auto dpuExec = nceOp.getSubExecutor(VPU::ExecutorKindAttr::get(module->getContext(), VPU::ExecutorKind::DPU));
    _NCEFrequency = nceOp.getProcessorFrequency().getValueAsDouble();
    _numClusters = nceOp.count();
    _numDPUs = dpuExec.count();
    _NCEThroughput = getNCEThroughput(VPU::getArch(nceOp));
    _DMABandwidth = getDMABandwidth(VPU::getArch(nceOp));
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
 * srcTensorType is the output of parent NCE op
 * dstTensorType is the input of child NCE op
 * return spilling write cost and spilling read cost
 */
LayerCostModel::SpillingCost LayerCostModel::getSpillingCost(vpux::NDTypeInterface srcTensorType,
                                                             vpux::NDTypeInterface dstTensorType) const {
    auto distributedSrcType = srcTensorType.dyn_cast<DistributedTensorType>();
    auto distributedDstType = dstTensorType.dyn_cast<DistributedTensorType>();
    auto srcMode = distributedSrcType != nullptr ? distributedSrcType.getDistribution().mode().getValue()
                                                 : VPU::DistributionMode::NONE;
    auto dstMode = distributedDstType != nullptr ? distributedDstType.getDistribution().mode().getValue()
                                                 : VPU::DistributionMode::NONE;

    auto spillingWriteCostFunc = spillingCostMap.at(srcMode);
    auto spillingReadCostFunc = spillingCostMap.at(dstMode);
    return {spillingWriteCostFunc(srcTensorType, SpillingType::SPILL_WRITE, _DDRLatency, _DMABandwidth, _CMXLatency,
                                  _CMXMulticastBandwidth),
            spillingReadCostFunc(dstTensorType, SpillingType::SPILL_READ, _DDRLatency, _DMABandwidth, _CMXLatency,
                                 _CMXMulticastBandwidth)};
}

double LayerCostModel::getSpillingReadCost(vpux::NDTypeInterface srcTensorType) const {
    auto distributedSrcType = srcTensorType.dyn_cast<DistributedTensorType>();
    auto srcMode = distributedSrcType != nullptr ? distributedSrcType.getDistribution().mode().getValue()
                                                 : VPU::DistributionMode::NONE;
    auto spillingReadCostFunc = spillingCostMap.at(srcMode);
    return spillingReadCostFunc(srcTensorType, SpillingType::SPILL_READ, _DDRLatency, _DMABandwidth, _CMXLatency,
                                _CMXMulticastBandwidth);
}

double LayerCostModel::getSpillingWriteCost(vpux::NDTypeInterface srcTensorType) const {
    auto distributedSrcType = srcTensorType.dyn_cast<DistributedTensorType>();
    auto srcMode = distributedSrcType != nullptr ? distributedSrcType.getDistribution().mode().getValue()
                                                 : VPU::DistributionMode::NONE;
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
    if (mlir::isa<VPU::NCEConvolutionOp>(op)) {
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
    const int64_t IC = mlir::isa<VPU::NCEConvolutionOp>(op) ? getShape(op->getOperand(0))[Dims4D::Act::C] : 1;
    const int64_t OC = getShape(op->getResult(0))[Dims4D::Act::C];
    const auto kernelSize = nceOp.getKernelSize();
    auto numClustersAttr = VPU::getOptimalNumClusters(clusteredOp, OC, strategy);

    /// Weights cost
    /// Weights and weightTable are Segmented mode under SOK (only including ddr -> cmx cost),
    /// SOK may use less clusters to avoid alignment
    /// So it's not proper to estimate total weightsSize by "clusterWeightsSize * _numClusters" simply
    /// Using distributed tensor for SOK to get accurate total size
    if (mlir::isa<VPU::NCEConvolutionOp>(op) || mlir::isa<VPU::NCEDepthConvolutionOp>(op)) {
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
                auto tiledWeightsShapes = distributedWeightsType.getPerClusterComputeShapes();
                auto tiledWeightsOffsets = distributedWeightsType.getPerClusterComputeShapeOffsets();

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
                            const auto alignedTileByteSize = vpux::alignVal<int64_t>(tileByteSize, weightSetAlignment);
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
                    clusterWeightsSize += alignVal<int64_t>(numElems * elemByteSize, _cmxAddressAlignment);
                }
            } else {
                clusterWeightsSize += (OC * alignVal<int64_t>(weightSetSize * elemByteSize, _cmxAddressAlignment));
            }

            if (weights.getType().isa<VPU::SparseTensorType>()) {
                const int64_t weightSetBitAlignment = 128;
                const int64_t sparsityMapSize = (OC * alignVal<int64_t>(weightSetSize, weightSetBitAlignment));
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
    if (mlir::isa<VPU::NCEConvolutionOp>(op) || mlir::isa<VPU::NCEDepthConvolutionOp>(op) ||
        mlir::isa<VPU::NCEMaxPoolOp>(op)) {
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
double LayerCostModel::getLayerCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy,
                                    bool useTimeBasedCost) const {
    if (!useTimeBasedCost) {
        return getEfficiencyCost(nceOp, strategy);
    }

    auto basicDPUandDMACost = getDPUandDMATimeCost(nceOp, strategy);
    _log.trace("Operation {0} get DPUandDMATimeCost {1} with strategy {2}", nceOp->getLoc(), basicDPUandDMACost,
               strategy);

    return basicDPUandDMACost;
}

/// @brief Time-cost : DPU time + DMA time (cycles)
/// @details DPU time calculation also considers the impact of DPU computing efficiency
double LayerCostModel::getDPUandDMATimeCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
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

bool LayerCostModel::hasSpilling(VPU::ClusteredOpInterface clustered, VPU::ClusteredOpInterface userOp) const {
    auto targetOutputTypes = hasMultiClusterStrategy(clustered)
                                     ? getDistributedOutputType(clustered, getMultiClusterStrategyValue(clustered))
                                               .cast<vpux::NDTypeInterface>()
                                     : getNormalOutputType(clustered);

    auto targetInputTypes = hasMultiClusterStrategy(userOp)
                                    ? getDistributedInputType(userOp, clustered, getMultiClusterStrategyValue(userOp))
                                              .cast<vpux::NDTypeInterface>()
                                    : getNormalInputType(userOp, clustered);
    return hasSpilling(clustered, targetOutputTypes, targetInputTypes);
}

bool LayerCostModel::hasSpilling(VPU::ClusteredOpInterface origOp, VPU::MultiClusterStrategy origOpStrategy,
                                 VPU::ClusteredOpInterface userOp) const {
    auto targetOutputTypes = getDistributedOutputType(origOp, origOpStrategy).cast<vpux::NDTypeInterface>();
    auto targetInputTypes = hasMultiClusterStrategy(userOp)
                                    ? getDistributedInputType(userOp, origOp, getMultiClusterStrategyValue(origOp))
                                              .cast<vpux::NDTypeInterface>()
                                    : getNormalInputType(userOp, origOp);
    return hasSpilling(origOp, targetOutputTypes, targetInputTypes);
}

bool LayerCostModel::hasSpilling(VPU::ClusteredOpInterface origOp, VPU::ClusteredOpInterface userOp,
                                 VPU::MultiClusterStrategy userOpStrategy) const {
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    auto targetOutputTypes = hasMultiClusterStrategy(origOp)
                                     ? getDistributedOutputType(origOp, getMultiClusterStrategyValue(clusteredOp))
                                               .cast<vpux::NDTypeInterface>()
                                     : getNormalOutputType(origOp);
    auto targetInputTypes = getDistributedInputType(userOp, origOp, userOpStrategy).cast<vpux::NDTypeInterface>();
    return hasSpilling(origOp, targetOutputTypes, targetInputTypes);
}

bool LayerCostModel::hasSpilling(VPU::ClusteredOpInterface origOp, VPU::MultiClusterStrategy origOpStrategy,
                                 VPU::ClusteredOpInterface userOp, VPU::MultiClusterStrategy userOpStrategy) const {
    auto targetOutputTypes = getDistributedOutputType(origOp, origOpStrategy).cast<vpux::NDTypeInterface>();
    auto targetInputTypes = getDistributedInputType(userOp, origOp, userOpStrategy).cast<vpux::NDTypeInterface>();
    return hasSpilling(origOp, targetOutputTypes, targetInputTypes);
}

bool LayerCostModel::doesLayerRequireTiling(VPU::ClusteredOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    auto layerStrategyChecker = LayerStrategyCheckerFactory::instance().get(nceOp->getName());
    return !(layerStrategyChecker->doesLayerFitIntoCMX(nceOp.getOperation(), strategy));
}

bool LayerCostModel::hasInputSpillingToMultiClusterLayer(VPU::ClusteredOpInterface origClusteredOp) const {
    SmallVector<mlir::Value> layerInputs;
    if (auto eltwiseOp = mlir::dyn_cast<VPU::NCEEltwiseOp>(origClusteredOp.getOperation())) {
        layerInputs.push_back(eltwiseOp.input1());
        layerInputs.push_back(eltwiseOp.input2());
    } else {
        layerInputs.push_back(origClusteredOp->getOperand(0));
    }

    for (auto input : layerInputs) {
        if (input.isa<mlir::BlockArgument>()) {
            continue;
        }
        auto parent = input.getDefiningOp();
        if (mlir::isa<VPU::ShapeCastOp, VPU::QuantizeCastOp>(parent)) {
            if (parent->getOperand(0).isa<mlir::BlockArgument>()) {
                continue;
            }
            // propagate cast ops
            parent = parent->getOperand(0).getDefiningOp();
        }
        if ((parent == nullptr) || (!hasMultiClusterStrategy(parent))) {
            continue;
        }

        if (auto parentClusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(parent)) {
            auto requireTiling =
                    doesLayerRequireTiling(parentClusteredOp, getMultiClusterStrategyValue(parentClusteredOp)) ||
                    doesLayerRequireTiling(origClusteredOp, getMultiClusterStrategyValue(origClusteredOp));
            return requireTiling || hasSpilling(parentClusteredOp, origClusteredOp);
        }
    }

    return false;
}

bool LayerCostModel::hasOutputSpillingToMultiClusterLayer(VPU::ClusteredOpInterface origOp) const {
    for (auto* userOp : origOp->getResult(0).getUsers()) {
        if (mlir::isa<VPU::QuantizeCastOp>(userOp) || (mlir::isa<VPU::ShapeCastOp>(userOp) && userOp->hasOneUse())) {
            // propagate cast ops
            userOp = *userOp->getResult(0).getUsers().begin();
        }
        if (mlir::isa<VPU::ClusteredOpInterface>(userOp) && hasSpilling(origOp, userOp)) {
            return true;
        }
    }
    return false;
}

LayerCostModel::SpillingCost LayerCostModel::calculateSpillingCost(VPU::ClusteredOpInterface parentOp,
                                                                   VPU::ClusteredOpInterface userOp,
                                                                   VPU::MultiClusterStrategy parentStrategy,
                                                                   VPU::MultiClusterStrategy userStrategy) const {
    auto targetOutputType = getDistributedOutputType(parentOp, parentStrategy);
    auto targetInputType = getDistributedInputType(userOp, parentOp, userStrategy);

    auto requireTiling =
            doesLayerRequireTiling(parentOp, parentStrategy) || doesLayerRequireTiling(userOp, userStrategy);

    if (requireTiling || hasSpilling(parentOp, targetOutputType, targetInputType)) {
        return getSpillingCost(targetOutputType, targetInputType);
    }
    return {0.0, 0.0};
}

// Return all output spillings from origOp with a given strategy to layers have multi-cluster strategy
double LayerCostModel::getOutputSpillingCostToMultiClusterLayer(VPU::ClusteredOpInterface clusteredOp,
                                                                VPU::MultiClusterStrategy strategy) const {
    bool hasCalculatedSpillingWriteCost = false;
    double totalSpillingCost = 0.0;
    for (auto user : clusteredOp->getResult(0).getUsers()) {
        if (mlir::isa<VPU::ShapeCastOp, VPU::QuantizeCastOp>(user)) {
            // propagate cast ops
            user = *user->getResult(0).getUsers().begin();
        }

        if (!hasMultiClusterStrategy(user)) {
            continue;
        }

        if (auto userOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(user)) {
            auto spillingCost =
                    calculateSpillingCost(clusteredOp, userOp, strategy, getMultiClusterStrategyValue(userOp));
            if (!hasCalculatedSpillingWriteCost) {
                totalSpillingCost += spillingCost.writeCost;
                hasCalculatedSpillingWriteCost = true;
            }
            totalSpillingCost += spillingCost.readCost;
        }
    }

    return totalSpillingCost;
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
    auto nceOp = mlir::dyn_cast<NCEOpInterface>(clusteredOp.getOperation());
    const auto isCompressConv = VPU::NCEInvariant::isCompressConvolution(arch, clusteredOp);
    if (clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverHeight) &&
        layerStrategy->isOperationSplitOverHeightCompatible(clusteredOp)) {
        splitOverHeightCost = getLayerCost(nceOp, VPU::MultiClusterStrategy::SplitOverHeight);
        _log.trace("SplitOverHeight cost is {0}", splitOverHeightCost);
        splitOverHeightFitIntoCMX =
                layerStrategy->doesLayerFitIntoCMX(clusteredOp, VPU::MultiClusterStrategy::SplitOverHeight);
    }

    if (clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverKernel) &&
        layerStrategy->isOperationSplitOverKernelCompatible(clusteredOp)) {
        splitOverKernelCost = getLayerCost(nceOp, VPU::MultiClusterStrategy::SplitOverKernel);
        _log.trace("splitOverKernel cost is {0}", splitOverKernelCost);
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
            nceOp->getResult(0).getType().template cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C];
    const auto sokOptimalClusters = getNumberOfClustersForSOKToAvoidAlignment(outputChannels, _numClusters);

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
    auto nceOpStrategy = LayerStrategyCheckerFactory::instance()._nceOpStrategies.find(name);
    VPUX_THROW_WHEN(nceOpStrategy == LayerStrategyCheckerFactory::instance()._nceOpStrategies.end(),
                    "Can not find NCE op strategies for Op: {0}", name);
    return nceOpStrategy->second;
}

LayerStrategyCheckerFactory& LayerStrategyCheckerFactory::instance() {
    static LayerStrategyCheckerFactory obj;
    return obj;
}

void LayerStrategyCheckerFactory::registerNCEOpStrategy(mlir::FuncOp func, vpux::Logger _log) {
    _nceOpStrategies[mlir::OperationName(NCEConvolutionOp::getOperationName(), func->getContext())] =
            std::make_shared<ConvolutionStrategy>(func, _log);
    _nceOpStrategies[mlir::OperationName(NCEDepthConvolutionOp::getOperationName(), func->getContext())] =
            std::make_shared<DepthConvolutionStrategy>(func, _log);
    _nceOpStrategies[mlir::OperationName(NCEMaxPoolOp::getOperationName(), func->getContext())] =
            std::make_shared<MaxPoolStrategy>(func, _log);
    _nceOpStrategies[mlir::OperationName(NCEAveragePoolOp::getOperationName(), func->getContext())] =
            std::make_shared<AveragePoolStrategy>(func, _log);
    _nceOpStrategies[mlir::OperationName(NCEEltwiseOp::getOperationName(), func->getContext())] =
            std::make_shared<EltwiseStrategy>(func, _log);
    _nceOpStrategies[mlir::OperationName(VPU::TanhOp::getOperationName(), func->getContext())] =
            std::make_shared<SWStrategy>(func, _log);
    _nceOpStrategies[mlir::OperationName(VPU::MVNOp::getOperationName(), func->getContext())] =
            std::make_shared<SWStrategy>(func, _log);
}
