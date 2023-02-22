//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/dialect/VPU/layer_strategy.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/dense_map.hpp"

namespace vpux {
namespace VPU {

class LayerStrategyCheckerFactory final {
public:
    BaseLayerStrategy::Ptr get(mlir::OperationName name);
    static LayerStrategyCheckerFactory& instance();
    void registerNCEOpStrategy(mlir::FuncOp func, vpux::Logger log);

private:
    LayerStrategyCheckerFactory() = default;
    LayerStrategyCheckerFactory(const LayerStrategyCheckerFactory&) = delete;
    LayerStrategyCheckerFactory& operator=(const LayerStrategyCheckerFactory&) = delete;

    DenseMap<mlir::OperationName, BaseLayerStrategy::Ptr> _nceOpStrategies;
};

//
// LayerCostModel for layer cost estimation given by different strategies
//
class LayerCostModel final {
public:
    struct SpillingCost {
        double writeCost;
        double readCost;
    };

    explicit LayerCostModel(mlir::FuncOp func, Logger log);
    ~LayerCostModel() = default;

    double getLayerCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy,
                        bool useTimeBasedCost = true) const;
    double getDPUandDMATimeCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const;
    double getEfficiencyCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const;

    bool hasMultiClusterStrategy(mlir::Operation* op) const;
    VPU::MultiClusterStrategy getMultiClusterStrategyValue(VPU::ClusteredOpInterface clusteredOp) const;
    bool hasSpilling(VPU::ClusteredOpInterface origOp, vpux::NDTypeInterface srcTensorType,
                     vpux::NDTypeInterface dstTensorType) const;
    bool hasSpilling(VPU::ClusteredOpInterface origOp, VPU::ClusteredOpInterface userOp) const;
    bool hasSpilling(VPU::ClusteredOpInterface origOp, VPU::MultiClusterStrategy origOpStrategyAttr,
                     VPU::ClusteredOpInterface userOp) const;
    bool hasSpilling(VPU::ClusteredOpInterface origOp, VPU::ClusteredOpInterface userOp,
                     VPU::MultiClusterStrategy userOpStrategyAttr) const;
    bool hasSpilling(VPU::ClusteredOpInterface origOp, VPU::MultiClusterStrategy origOpStrategy,
                     VPU::ClusteredOpInterface userOp, VPU::MultiClusterStrategy userOpStrategy) const;
    bool doesLayerRequireTiling(VPU::ClusteredOpInterface nceOp, VPU::MultiClusterStrategy strategy) const;
    bool hasInputSpillingToMultiClusterLayer(VPU::ClusteredOpInterface origOp) const;
    bool hasOutputSpillingToMultiClusterLayer(VPU::ClusteredOpInterface origOp) const;
    double getOutputSpillingCostToMultiClusterLayer(VPU::ClusteredOpInterface origOp,
                                                    VPU::MultiClusterStrategy strategy) const;
    double getSpillingReadCost(vpux::NDTypeInterface srcTensorType) const;
    double getSpillingWriteCost(vpux::NDTypeInterface srcTensorType) const;
    SpillingCost calculateSpillingCost(VPU::ClusteredOpInterface parentOp, VPU::ClusteredOpInterface userOp,
                                       VPU::MultiClusterStrategy parentStrategy,
                                       VPU::MultiClusterStrategy userStrategy) const;
    vpux::NDTypeInterface getNormalInputType(VPU::ClusteredOpInterface origOp, mlir::Operation* parentOp) const;
    vpux::NDTypeInterface getNormalOutputType(VPU::ClusteredOpInterface origOp) const;
    VPU::DistributedTypeInterface getDistributedInputType(VPU::ClusteredOpInterface origOp, mlir::Operation* parentOp,
                                                          VPU::MultiClusterStrategy specifiedStrategy) const;
    VPU::DistributedTypeInterface getDistributedOutputType(VPU::ClusteredOpInterface origOp,
                                                           VPU::MultiClusterStrategy specifiedStrategy) const;

    VPU::MultiClusterStrategy getOptimalLayerStrategy(VPU::ClusteredOpInterface nceOp,
                                                      BaseLayerStrategy::Ptr layerStrategy) const;
    double COST_MAX = std::numeric_limits<double>::infinity();

private:
    double calculateMPEVolume(VPU::MPEMode mpeMode, Shape shape) const;

    double computeSplitEfficiency(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const;
    double clusterComputeTime(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const;
    double totalDMATime(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const;
    SpillingCost getSpillingCost(vpux::NDTypeInterface srcTensorType, vpux::NDTypeInterface dstTensorType) const;
    // Note that the cost model will be integrated with VPUNN as part of  E#37379
    // after that these platform specific variables will be removed
    double _NCEThroughput;                     // NCE throughput for U8 data, unit is MOPS
    const double _DDRLatency = 100;            // DDR latency is ~100 cycles per dma
    double _DMABandwidth;                      // Transition Bytes per cycle
    const double _CMXLatency = 5;              // Cycles, attempt to capture cost accessing CMX
    const double _CMXMulticastBandwidth = 32;  // 32 Bytes per cycle for multicast
    double _NCEFrequency;                      // NCE frequency, unit is MHz
    const int64_t _numChannelAlignment = 16;
    const int64_t _cmxAddressAlignment = 16;  // Kernel address alignment
    int64_t _numClusters;
    int64_t _numDPUs;
    mlir::FuncOp _func;
    Logger _log;
};

}  // namespace VPU
}  // namespace vpux
