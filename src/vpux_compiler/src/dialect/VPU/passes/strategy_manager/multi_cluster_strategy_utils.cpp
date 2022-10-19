//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <llvm/ADT/TypeSwitch.h>
#include <unordered_map>
#include "vpux/compiler/dialect/VPU/strategy_manager.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"

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
    const int64_t elemBytes = Byte(tensorType.getElemTypeSize()).count();
    int64_t totalSize = 0;
    // sums up of all the sub tensors to get the total size
    for (auto& shape : shapes) {
        totalSize += shape.totalSize() * elemBytes;
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

mlir::Value getInputFromNceOp(VPU::NCEOpInterface nceOp, mlir::Operation* parentOp) {
    for (auto operand : nceOp->getOperands()) {
        if (operand.getDefiningOp() == parentOp) {
            return operand;
        }
    }
    VPUX_THROW("Cannot find input from op: {0}, parent op: {1}", nceOp, parentOp);
}

vpux::NDTypeInterface getTargetOutputType(VPU::NCEOpInterface origOp, mlir::Attribute specifiedStrategyAttr) {
    if (!specifiedStrategyAttr) {
        return origOp->getResult(0).getType().dyn_cast<vpux::NDTypeInterface>();
    }
    auto specifiedStrategy = specifiedStrategyAttr.cast<VPU::MultiClusterStrategyAttr>().getValue();
    auto numClustersAttr = VPU::getOptimalNumClusters(
            origOp, origOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C],
            specifiedStrategy);
    return VPU::getDistributedOutputTypeFromOp(origOp, origOp->getResult(0).getType(), numClustersAttr,
                                               specifiedStrategy);
}

vpux::NDTypeInterface getTargetInputType(VPU::NCEOpInterface origOp, mlir::Operation* parentOp,
                                         mlir::Attribute specifiedStrategyAttr) {
    auto input = getInputFromNceOp(origOp, parentOp);
    if (!specifiedStrategyAttr) {
        return input.getType().dyn_cast<vpux::NDTypeInterface>();
    }
    auto specifiedStrategy = specifiedStrategyAttr.cast<VPU::MultiClusterStrategyAttr>().getValue();
    auto numClustersAttr = VPU::getOptimalNumClusters(
            origOp, origOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C],
            specifiedStrategy);
    return getDistributedActivationTypeFromOp(origOp, input.getType(), numClustersAttr, specifiedStrategy);
}

bool isAlignmentCompatible(VPU::NCEOpInterface nceOp, vpux::NDTypeInterface srcType, vpux::NDTypeInterface dstType) {
    auto distributedSrcType = srcType.dyn_cast<DistributedTensorType>();
    auto distributedDstType = dstType.dyn_cast<DistributedTensorType>();
    if (!distributedSrcType || !distributedDstType) {
        return false;
    }

    const auto newDistributedOutputTensorType = adjustOutputAlignmentForSOH(nceOp, distributedDstType);
    if (newDistributedOutputTensorType.hasValue()) {
        return isDistributedCastCompatible(newDistributedOutputTensorType.getValue(), distributedDstType).succeeded();
    }

    return false;
}

bool isTargetTensorTypeCompatible(vpux::NDTypeInterface srcType, vpux::NDTypeInterface dstType) {
    auto distributedSrcType = srcType.dyn_cast<DistributedTensorType>();
    auto distributedDstType = dstType.dyn_cast<DistributedTensorType>();
    auto srcIsDistributed = distributedSrcType != nullptr;
    auto dstIsDistributed = distributedDstType != nullptr;
    if (!srcIsDistributed && !dstIsDistributed) {
        return true;
    }
    if (srcIsDistributed ^ dstIsDistributed) {
        return false;
    }
    return isDistributedCastCompatible(distributedSrcType, distributedDstType).succeeded();
}
}  // namespace

LayerCostModel::LayerCostModel(mlir::FuncOp func, Logger log): _func(func), _log(log) {
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto nceOp = IE::getAvailableExecutor(module, ExecutorKind::NCE);
    auto dpuExec = nceOp.getSubExecutor(VPU::ExecutorKindAttr::get(module->getContext(), VPU::ExecutorKind::DPU));
    _NCEFrequency = nceOp->getAttr(VPU::getProcessorFrequencyAttrName()).cast<mlir::FloatAttr>().getValueAsDouble();
    _numClusters = nceOp.count();
    _numDPUs = dpuExec.count();
}

/*
 * Get the spilling cost
 * srcTensorShape is the output of parent NCE op
 * dstTensorShape is the input of child NCE op
 * return spilling write cost and spilling read cost
 */
LayerCostModel::SpillingCost LayerCostModel::getSpillingCost(VPU::NCEOpInterface parentOp,
                                                             vpux::NDTypeInterface srcTensorType,
                                                             vpux::NDTypeInterface dstTensorType) const {
    if (!hasSpilling(parentOp, srcTensorType, dstTensorType)) {
        return {0.0, 0.0};
    }

    auto distributedSrcType = srcTensorType.dyn_cast<DistributedTensorType>();
    auto distributedDstType = dstTensorType.dyn_cast<DistributedTensorType>();
    auto srcMode = distributedSrcType != nullptr ? distributedSrcType.getDistribution().mode().getValue()
                                                 : VPU::DistributionMode::NONE;
    auto dstMode = distributedDstType != nullptr ? distributedDstType.getDistribution().mode().getValue()
                                                 : VPU::DistributionMode::NONE;

    auto spillingWriteCostFunc = spillingCostMap.at(srcMode);
    auto spillingReadCostFunc = spillingCostMap.at(dstMode);
    return {spillingWriteCostFunc(srcTensorType, SpillingType::SPILL_WRITE, _DDRLatency, _DDRBandwidth, _CMXLatency,
                                  _CMXMulticastBandwidth),
            spillingReadCostFunc(dstTensorType, SpillingType::SPILL_READ, _DDRLatency, _DDRBandwidth, _CMXLatency,
                                 _CMXMulticastBandwidth)};
}

/*
 * Get the input spilling cost with a given strategy (cycles)
 */
double LayerCostModel::getInputSpillingCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    return llvm::TypeSwitch<mlir::Operation*, double>(nceOp.getOperation())
            .Case<NCEMaxPoolOp>([&](NCEMaxPoolOp origOp) {
                return getInputSpillingCost(origOp, origOp.input(), strategy);
            })
            .Case<NCEAveragePoolOp>([&](NCEAveragePoolOp origOp) {
                return getInputSpillingCost(origOp, origOp.input(), strategy);
            })
            .Case<NCEEltwiseOp>([&](NCEEltwiseOp origOp) {
                return getInputSpillingCost(origOp, origOp.input1(), strategy) +
                       getInputSpillingCost(origOp, origOp.input2(), strategy);
            })
            .Case<NCEConvolutionOp>([&](NCEConvolutionOp origOp) {
                return getInputSpillingCost(origOp, origOp.input(), strategy);
            })
            .Case<NCEDepthConvolutionOp>([&](NCEDepthConvolutionOp origOp) {
                return getInputSpillingCost(origOp, origOp.input(), strategy);
            })
            .Default([](mlir::Operation*) {
                return 0.0;
            });
}

/*
 * Get the input spilling cost for specified input operand
 */
double LayerCostModel::getInputSpillingCost(VPU::NCEOpInterface nceOp, mlir::Value input,
                                            VPU::MultiClusterStrategy strategy) const {
    auto targetTensorType = getTargetInputType(nceOp, input.getDefiningOp(),
                                               VPU::MultiClusterStrategyAttr::get(nceOp->getContext(), strategy));

    if (input.getDefiningOp() == nullptr) {
        return getSpillingReadCost(targetTensorType);
    }

    return llvm::TypeSwitch<mlir::Operation*, double>(input.getDefiningOp())
            .Case<VPU::NCEOpInterface>([&](VPU::NCEOpInterface parentOp) {
                auto parentStrategyAttr =
                        parentOp->hasAttr(multiClusterStrategy)
                                ? parentOp->getAttr(multiClusterStrategy).dyn_cast<VPU::MultiClusterStrategyAttr>()
                                : nullptr;
                auto strategyAttr = VPU::MultiClusterStrategyAttr::get(nceOp->getContext(), strategy);
                auto currentSpillingCost = calculateSpillingCost(parentOp, nceOp, parentStrategyAttr, strategyAttr);
                return currentSpillingCost.writeCost + currentSpillingCost.readCost;
            })

            .Case<IE::ConcatOp>([&](IE::ConcatOp parentOp) {
                // NOTE: If the concat is a CMX concat and the outputs are duplicated in each cluster, there is
                // supposed to be no spilling when strategy is Clustering or SplitOverKernel. For now it's just
                // a workaround to check the strategy only since we don't know whether it's a cmx concat or not now.
                bool needSpilling = true;
                if (strategy == MultiClusterStrategy::SplitOverKernel || strategy == MultiClusterStrategy::Clustering) {
                    SmallVector<VPU::NCEOpInterface> nceInputOps;
                    for (auto concatInput : parentOp.inputs()) {
                        if (auto nceInput = concatInput.getDefiningOp<VPU::NCEOpInterface>()) {
                            nceInputOps.push_back(nceInput);
                        }
                    }
                    auto hasIncompatibleStrategy = [](VPU::NCEOpInterface nceInput) {
                        if (!nceInput->hasAttr(multiClusterStrategy)) {
                            return true;
                        }
                        auto nceInputStrategy = nceInput->getAttr(multiClusterStrategy)
                                                        .cast<VPU::MultiClusterStrategyAttr>()
                                                        .getValue();
                        return nceInputStrategy != MultiClusterStrategy::Clustering;
                    };
                    needSpilling = nceInputOps.empty() || llvm::any_of(nceInputOps, hasIncompatibleStrategy);
                }
                return needSpilling ? getSpillingReadCost(targetTensorType) : 0.0;
            })
            .Default([&](mlir::Operation*) {
                return getSpillingReadCost(targetTensorType);
            });
}

double LayerCostModel::getSpillingReadCost(vpux::NDTypeInterface srcTensorType) const {
    auto distributedSrcType = srcTensorType.dyn_cast<DistributedTensorType>();
    auto srcMode = distributedSrcType != nullptr ? distributedSrcType.getDistribution().mode().getValue()
                                                 : VPU::DistributionMode::NONE;
    auto spillingReadCostFunc = spillingCostMap.at(srcMode);
    return spillingReadCostFunc(srcTensorType, SpillingType::SPILL_READ, _DDRLatency, _DDRBandwidth, _CMXLatency,
                                _CMXMulticastBandwidth);
}

// The function computes the actual output tensor volume (i.e. computation that is performed)
// given the stratey and the MPE mode
double LayerCostModel::calculateMPEVolume(VPU::MPEMode mpeMode, Shape shape) const {
    int64_t mpeHeight;
    int64_t mpeWidth;
    if (mpeMode == VPU::MPEMode::VECTOR) {
        mpeHeight = 16;
        mpeWidth = 1;
    } else if (mpeMode == VPU::MPEMode::MATRIX) {
        mpeHeight = 4;
        mpeWidth = 4;
    } else {
        VPUX_THROW("Unsupported MPE mode {0}", mpeMode);
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
    VPUX_THROW_UNLESS(nceOp.checkStrategyCompatibility(strategy) == true,
                      "Unsupported multi-cluster strategy '{0}' for layer type '{1}'", strategy, nceOp->getName());
    const auto outputTensorDistributionMode = getOutputTensorDistributionMode(strategy);
    auto numClusters = getOptimalNumClusters(
            nceOp, nceOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C], strategy);
    const auto outputTensorNumTiles = getIntArrayAttr(
            nceOp->getContext(), getOutputTensorNumTiles(nceOp->getResult(0).getType().cast<vpux::NDTypeInterface>(),
                                                         numClusters.getInt(), strategy));
    mlir::ArrayAttr outputAlignmentAttr = nullptr;
    const auto outputAlignment = getOutputTensorAlignment(strategy);
    if (outputAlignment.hasValue()) {
        outputAlignmentAttr = getIntArrayAttr(nceOp->getContext(), outputAlignment.getValue());
    }
    const auto distributedOutputTensorType = createDistributedTensorType(
            nceOp, nceOp->getResult(0).getType().cast<vpux::NDTypeInterface>(), outputTensorDistributionMode,
            outputTensorNumTiles, numClusters, outputAlignmentAttr);

    const auto perClusterShape = distributedOutputTensorType.getLargestCompactShape();
    const auto perClusterOutputTensorVolume =
            perClusterShape[Dims4D::Act::H] * perClusterShape[Dims4D::Act::W] * perClusterShape[Dims4D::Act::C];

    return std::max(static_cast<double>(perClusterOutputTensorVolume) /
                            calculateMPEVolume(VPU::MPEMode::MATRIX, perClusterShape),
                    static_cast<double>(perClusterOutputTensorVolume) /
                            calculateMPEVolume(VPU::MPEMode::VECTOR, perClusterShape));
}

// Returns the duration in cycles for the execution of a NCE task
double LayerCostModel::clusterComputeTime(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    VPUX_THROW_UNLESS(nceOp.checkStrategyCompatibility(strategy) == true,
                      "Unsupported multi-cluster strategy '{0}' for layer type '{1}'", strategy, nceOp->getName());

    double clusterOpsPerCycle = _NCEThroughput / _NCEFrequency / _numClusters;
    double clusterEff = computeSplitEfficiency(nceOp, strategy);
    auto largestClusterOutShape = getLargestClusterOutputShape(nceOp, strategy);

    auto kernelSize = nceOp.getKernelSize();
    auto op = nceOp.getOperation();
    int64_t baseKernelCost = kernelSize[Dims4D::Kernel::Y.ind()] * kernelSize[Dims4D::Kernel::X.ind()];
    if (mlir::isa<VPU::NCEConvolutionOp>(op)) {
        int64_t IC = getShape(
                op->getOperand(0))[Dims4D::Act::C];  // Get input channel (already channel-alignment in previous pass)
        baseKernelCost = IC * baseKernelCost;

    } else if (mlir::isa<VPU::NCEMaxPoolOp>(op) || mlir::isa<VPU::NCEAveragePoolOp>(op) ||
               mlir::isa<VPU::NCEDepthConvolutionOp>(op)) {
        baseKernelCost = baseKernelCost;
    } else if (mlir::isa<VPU::NCEEltwiseOp>(op)) {
        baseKernelCost = 1;
    } else {
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
/// @details Data transfering cost is modeled as (latency + size / bandwidth)
double LayerCostModel::totalDMATime(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    VPUX_THROW_UNLESS(nceOp.checkStrategyCompatibility(strategy) == true,
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
    auto numClustersAttr = VPU::getOptimalNumClusters(nceOp, OC, strategy);

    /// Weights cost
    /// Weights and weightTable are Segmented mode under SOK (only including ddr -> cmx cost),
    /// SOK may use less clusters to avoid alignment
    /// So it's not proper to estimate total weightsSize by "clusterWeightsSize * _numClusters" simply
    /// Using distributed tensor for SOK to get accurate total size
    if (mlir::isa<VPU::NCEConvolutionOp>(op) || mlir::isa<VPU::NCEDepthConvolutionOp>(op)) {
        auto weights = op->getOperand(1);
        const int64_t elemBytes = getElemTypeSize(weights.getType()).count() / CHAR_BIT;

        if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
            auto distributedWeightsTensorType =
                    VPU::getDistributedFilterTypeFromOp(nceOp, weights.getType(), numClustersAttr, strategy);
            auto tiledWeightsShapes = distributedWeightsTensorType.getPerClusterComputeShapes();

            int64_t totalWeightsSize = 0;
            for (auto& clusterWeightsShape : tiledWeightsShapes) {
                totalWeightsSize += clusterWeightsShape.totalSize();
            }
            totalWeightsSize *= elemBytes;
            double weightCycles = static_cast<double>(totalWeightsSize) / _DDRBandwidth;
            totalWeightCycles = _DDRLatency + weightCycles;
        } else {
            // Duplicated mode in other strategies has ddr->cmx cost
            // IC * KX * KY * BytesPerElement needs to align to 16B for kernels
            int64_t clusterWeightsSize =
                    (OC * alignVal<int64_t>(IC * kernelSize[Dims4D::Kernel::Y.ind()] *
                                                    kernelSize[Dims4D::Kernel::X.ind()] * elemBytes,
                                            _cmxAddressAlignment));
            totalWeightCycles = _DDRLatency + (static_cast<double>(clusterWeightsSize) / _DDRBandwidth);
        }
    }

    /// WeightsTable cost
    /// WeightTable has OC entries, each entry includes sparsity/weights pointer, bias and mult/shfit quantized
    /// params. The total size for each entry is 16 Bytes
    if (mlir::isa<VPU::NCEConvolutionOp>(op) || mlir::isa<VPU::NCEDepthConvolutionOp>(op) ||
        mlir::isa<VPU::NCEMaxPoolOp>(op)) {
        auto largestClusterOutShape = getLargestClusterOutputShape(nceOp, strategy);
        int64_t alignedClusterOutChannels = largestClusterOutShape[Dims4D::Act::C];
        int64_t clusterWeightTableSize = NCEInvariant::getWeightsTableSize(alignedClusterOutChannels).count();

        if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
            totalWeightsTableCycles =
                    _DDRLatency +
                    static_cast<double>(clusterWeightTableSize * numClustersAttr.getInt()) / _DDRBandwidth;
        } else {
            totalWeightsTableCycles = _DDRLatency + (static_cast<double>(clusterWeightTableSize) / _DDRBandwidth);
        }
    }

    /// ActivationWindow cost
    /// It's always duplicated mode and only dwconv , cmconv and maxpool own it
    int64_t activationWindowSize = 0;
    if (mlir::isa<VPU::NCEMaxPoolOp>(op) || mlir::isa<VPU::NCEDepthConvolutionOp>(op) ||
        (mlir::isa<VPU::NCEConvolutionOp>(op) && isCMajor)) {
        const auto SX = Shape(nceOp.getStrides())[Dims4D::Strides::X];
        const auto inputElemType = op->getOperand(0).getType().cast<mlir::ShapedType>().getElementType();
        auto sparsityMode = VPU::NCESparsity::Mode::POOL;
        if (mlir::isa<VPU::NCEConvolutionOp>(op)) {
            sparsityMode = VPU::NCESparsity::Mode::CM_CONV;
        } else if (mlir::isa<VPU::NCEDepthConvolutionOp>(op)) {
            sparsityMode = VPU::NCESparsity::Mode::DW_CONV;
        }

        activationWindowSize =
                VPU::NCESparsity::getActivationWindowSize(sparsityMode, Shape(kernelSize), SX, inputElemType, IC);
    }
    totalActivationWindowCycles = _DDRLatency + (static_cast<double>(activationWindowSize) / _DDRBandwidth);

    /// PWLTable cost
    /// DistributionMode is Duplicated, at most we have 32 instructions (8 segments for pwl),
    /// so size is 32 x INT32 = 128 Bytes
    if (auto origOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(op)) {
        if (origOp.instructionListTable()) {
            pwlTableCycles = _DDRLatency + (128.0 / _DDRBandwidth);
        }
    } else if (auto origOp = mlir::dyn_cast<VPU::NCEDepthConvolutionOp>(op)) {
        if (origOp.instructionListTable()) {
            pwlTableCycles = _DDRLatency + (128.0 / _DDRBandwidth);
        }
    }

    /// @brief This section captures the output tensors to multicast to other clusters
    /// @details ODU multicast only happens in SOK and HKSwitch strategy,
    /// Multicast cost can be modeled as one cluster emit its data to other all clusters in parallel (multicast)
    /// and each multicast is executed in series
    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel || strategy == VPU::MultiClusterStrategy::HKSwitch) {
        const int64_t outElemBytes = getElemTypeSize(op->getResult(0).getType()).count() / CHAR_BIT;
        int64_t outputSize = 0;
        auto tiledOutShapes = getPerClusterOutputShape(nceOp, strategy);
        for (auto& clusterOutShape : tiledOutShapes) {
            outputSize += clusterOutShape.totalSize();
        }
        outputCycles += ((static_cast<double>(_numClusters) * _CMXLatency) +
                         (static_cast<double>(outputSize * outElemBytes) / _CMXMulticastBandwidth));
    }

    // Total cost for single layer
    return totalActivationWindowCycles + totalWeightCycles + totalWeightsTableCycles + pwlTableCycles + outputCycles;
}

/// @brief A switcher to select time-cost or efficiency-cost for greedy strategy assignment
/// @details Time-cost includes an extra input spilling cost to be more accurate
double LayerCostModel::getLayerCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy,
                                    bool useTimeBasedCost) const {
    return useTimeBasedCost ? (getDPUandDMATimeCost(nceOp, strategy) + getInputSpillingCost(nceOp, strategy))
                            : getEfficiencyCost(nceOp, strategy);
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

/// Anywhere if you need to judge spilling existing, please call me!
/// srcTensorType is the output of parent origOp
/// dstTensorType is the input of child NCE op
bool LayerCostModel::hasSpilling(VPU::NCEOpInterface origOp, vpux::NDTypeInterface srcTensorType,
                                 vpux::NDTypeInterface dstTensorType) const {
    if (isTargetTensorTypeCompatible(srcTensorType, dstTensorType) ||
        isAlignmentCompatible(origOp, srcTensorType, dstTensorType)) {
        return false;
    }
    return true;
}

bool LayerCostModel::hasSpilling(VPU::NCEOpInterface origOp, VPU::NCEOpInterface userOp) const {
    auto targetOutputType = getTargetOutputType(origOp, origOp->getAttr(multiClusterStrategy));
    auto targetInputType = getTargetInputType(userOp, origOp, userOp->getAttr(multiClusterStrategy));
    return hasSpilling(origOp, targetOutputType, targetInputType);
}

bool LayerCostModel::hasSpilling(VPU::NCEOpInterface origOp, mlir::Attribute origOpStrategyAttr,
                                 VPU::NCEOpInterface userOp) const {
    auto targetOutputType = getTargetOutputType(origOp, origOpStrategyAttr);
    auto targetInputType = getTargetInputType(userOp, origOp, userOp->getAttr(multiClusterStrategy));
    return hasSpilling(origOp, targetOutputType, targetInputType);
}

bool LayerCostModel::hasOutputSpilling(VPU::NCEOpInterface origOp) const {
    return llvm::any_of(origOp->getResult(0).getUsers(), [&](mlir::Operation* userOp) {
        return mlir::isa<VPU::NCEOpInterface>(userOp) && hasSpilling(origOp, userOp);
    });
}

// Return all output spillings of origOp with a given strategy
double LayerCostModel::getOutputSpillingCost(VPU::NCEOpInterface origOp, VPU::MultiClusterStrategy strategy) const {
    SmallVector<vpux::NDTypeInterface> targetInputTensors;
    auto strategyAttr = VPU::MultiClusterStrategyAttr::get(origOp->getContext(), strategy);
    for (auto user : origOp->getResult(0).getUsers()) {
        if (mlir::isa<VPU::NCEOpInterface>(user) && hasSpilling(origOp, strategyAttr, user)) {
            auto targetInputType = getTargetInputType(user, origOp, user->getAttr(multiClusterStrategy));
            targetInputTensors.push_back(targetInputType);
        }
    }

    bool hasCalculatedSpillingWriteCost = false;
    double totalSpillingCost = 0.0;
    auto targetOutputType = getTargetOutputType(origOp, strategyAttr);
    for (auto& inputTensor : targetInputTensors) {
        auto spillingCost = getSpillingCost(origOp, targetOutputType, inputTensor);
        if (!hasCalculatedSpillingWriteCost) {
            totalSpillingCost += spillingCost.writeCost;
            hasCalculatedSpillingWriteCost = true;
        }
        totalSpillingCost += spillingCost.readCost;
    }

    return totalSpillingCost;
}

LayerCostModel::SpillingCost LayerCostModel::calculateSpillingCost(
        VPU::NCEOpInterface parentOp, VPU::NCEOpInterface userOp, VPU::MultiClusterStrategyAttr parentStrategyAttr,
        VPU::MultiClusterStrategyAttr userStrategyAttr) const {
    auto targetOutputType = getTargetOutputType(parentOp, parentStrategyAttr);
    auto targetInputType = getTargetInputType(userOp, parentOp, userStrategyAttr);
    return getSpillingCost(parentOp, targetOutputType, targetInputType);
}

VPU::MultiClusterStrategy LayerCostModel::getOptimalLayerStrategy(VPU::NCEOpInterface nceOp,
                                                                  BaseLayerStrategy::Ptr layerStrategy) const {
    double splitOverHeightCost = COST_MAX;
    double splitOverKernelCost = COST_MAX;
    const auto arch = VPU::getArch(nceOp);
    const auto isChannelMajor = (DimsOrder::fromValue(nceOp->getOperand(0)) == DimsOrder::NCHW) &&
                                VPU::NCEInvariant::isChannelMajorCompatible(
                                        arch, nceOp->getOperand(0).getType().cast<vpux::NDTypeInterface>());

    if (layerStrategy->isOperationSplitOverHeightCompatible(nceOp)) {
        splitOverHeightCost = getLayerCost(nceOp, VPU::MultiClusterStrategy::SplitOverHeight);
    }

    if (layerStrategy->isOperationSplitOverKernelCompatible(nceOp)) {
        splitOverKernelCost = getLayerCost(nceOp, VPU::MultiClusterStrategy::SplitOverKernel);
    }

    // Compute amount of clusters so that SOK is compatible
    const auto outputChannels =
            nceOp->getResult(0).getType().template cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C];
    const auto sokOptimalClusters = getNumberOfClustersForSOKToAvoidAlignment(outputChannels, _numClusters);
    const auto optimalHeightTiling = [&](void) {
        return isChannelMajor ? VPU::MultiClusterStrategy::SplitOverHeightOverlapped
                              : VPU::MultiClusterStrategy::SplitOverHeight;
    };

    if (sokOptimalClusters == _numClusters) {
        if (splitOverHeightCost <= splitOverKernelCost) {
            return optimalHeightTiling();
        }
        return VPU::MultiClusterStrategy::SplitOverKernel;
    }
    // SOH is P1 option as SOK may not utilize full clusters
    // However, it is still more optimal than clustering
    if (splitOverHeightCost != COST_MAX) {
        return optimalHeightTiling();
    }
    return VPU::MultiClusterStrategy::SplitOverKernel;
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
}
