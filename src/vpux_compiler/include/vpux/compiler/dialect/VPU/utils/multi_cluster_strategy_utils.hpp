//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/dialect/VPU/layer_strategy.hpp"
#include "vpux/compiler/dialect/VPU/utils.hpp"
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

    bool hasSpilling(VPU::NCEOpInterface origOp, vpux::NDTypeInterface srcTensorType,
                     vpux::NDTypeInterface dstTensorType) const;
    bool hasSpilling(VPU::NCEOpInterface origOp, VPU::NCEOpInterface userOp) const;
    bool hasSpilling(VPU::NCEOpInterface origOp, mlir::Attribute origOpStrategyAttr, VPU::NCEOpInterface userOp) const;
    bool hasOutputSpilling(VPU::NCEOpInterface origOp) const;
    double getOutputSpillingCost(VPU::NCEOpInterface origOp, VPU::MultiClusterStrategy strategy) const;
    double getInputSpillingCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const;
    SpillingCost calculateSpillingCost(VPU::NCEOpInterface parentOp, VPU::NCEOpInterface userOp,
                                       VPU::MultiClusterStrategyAttr parentStrategyAttr,
                                       VPU::MultiClusterStrategyAttr userStrategyAttr) const;

    VPU::MultiClusterStrategy getOptimalLayerStrategy(VPU::NCEOpInterface nceOp,
                                                      BaseLayerStrategy::Ptr layerStrategy) const;
    double COST_MAX = std::numeric_limits<double>::infinity();

private:
    double calculateMPEVolume(VPU::MPEMode mpeMode, Shape shape) const;

    double computeSplitEfficiency(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const;
    double clusterComputeTime(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const;
    double totalDMATime(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const;
    SpillingCost getSpillingCost(VPU::NCEOpInterface parentOp, vpux::NDTypeInterface srcTensorType,
                                 vpux::NDTypeInterface dstTensorType) const;
    double getInputSpillingCost(VPU::NCEOpInterface nceOp, mlir::Value, VPU::MultiClusterStrategy strategy) const;
    double getSpillingReadCost(vpux::NDTypeInterface srcTensorType) const;
    // Hardware parameters specified in VPUX30XX data book v1.2
    // Note that the cost model will be integrated with VPUNN as part of  E#37379
    // after which these platform specific variables with be removed
    const double _NCEThroughput = 7168000;     // VPUX30XX NCE throughput is 7.168 TOPS for U8 data
    const double _DDRLatency = 100;            // DDR latency is ~100 cycles per dma
    const double _DDRBandwidth = 28.6;         // (20000 MB/s) / 700 MHZ Bytes per cycle
    const double _CMXLatency = 5;              // Cycles, attempt to capture cost accessing CMX
    const double _CMXMulticastBandwidth = 32;  // 32 Bytes per cycle for multicast
    double _NCEFrequency;                      // 700 MHZ for KMB
    const int64_t _numChannelAlignment = 16;
    const int64_t _cmxAddressAlignment = 16;  // Kernel address alignment
    int64_t _numClusters;
    int64_t _numDPUs;
    mlir::FuncOp _func;
    Logger _log;
};

}  // namespace VPU
}  // namespace vpux
