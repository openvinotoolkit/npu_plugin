//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include <map>
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/utils/logging.hpp"
namespace vpux {
namespace VPU {

//
// LayerCostModel for layer time estimation given by different strategies
//
class LayerCostModel final {
public:
    explict LayerCostModel(mlir::FuncOp func, Logger log);
    ~LayerCostModel() = default;

    template <class ConcreteOp>
    double getLayerCost(ConcreteOp op, StringRef strategy, bool isTimeCost);
    // The recommended cost: dpu + dma time cost,
    // actually dpu time also includes below dpu efficiency
    template <class ConcreteOp>
    double getTimeCost(ConcreteOp op, StringRef strategy);
    // A simple cost only considering dpu efficiency
    template <class ConcreteOp>
    double getEfficiencyCost(ConcreteOp op, StringRef strategy);

privated:
    double calculateMPEVolume(VPU::MPEMode mpeMode, Shape shape) const;
    template <class ConcreteOp>
    double computeSplitEfficiency(ConcreteOp op, StringRef strategy) const;
    template <class ConcreteOp>
    double clusterComputeTime(ConcreteOp op, MultiClusterStrategy Strategy) const;
    template <class ConcreteOp>
    double dmaTime(ConcreteOp op, MultiClusterStrategy Strategy) const;
    
    // @warning Here exists a duplicated _numClusters with BaseLayerStrategy,
    // do we have a better way to access it without redefining? 
    // I cann't use friend class feature as BaseLayerStrategy is abstract class
    int64_t _numClusters;
    double _CMXBandwidth;
    double _DDRBandwidth;
    double _CMXLatency;
    double _DDRLatency;
    const size_t _cmxAddressAlignment = 16;  // This one for kernel address alignment
    mlir::FuncOp _func;
    Logger _log;
}

//
// BaseLayerStrategy
//

// Abstract base class
// Specific method implementations for each layer type are required
// in the derived classes
// Examples:
// (1) Does a particular layer with a particular strategy fit in CMX
// (2) The hardware efficiency for a particular layer with a particular strategy
//
// Note: This will probably be replaced by operation interface for operation
// cost model EISW-26043.
class BaseLayerStrategy {
public:
    explicit BaseLayerStrategy(mlir::FuncOp func, Logger log);
    virtual ~BaseLayerStrategy() = default;

    virtual bool isOperationSplitOverHeightCompatible(mlir::Operation* op) const;
    virtual bool isOperationSplitOverKernelCompatible(mlir::Operation* op) const;
    virtual bool isOperationMultiClusterCompatible(mlir::Operation* op) const;

    template <class ConcreteOp>
    StringRef getOptimalLayerStrategy(ConcreteOp op) const;

protected:
    virtual bool doesLayerFitIntoCMX(mlir::Operation* op, StringRef strategy) const = 0;

protected:
    int64_t _numClusters;
    int64_t _numDPUs;
    int64_t _minimumOutputHeightForSOH;
    const int64_t _numChannelAlignment = 16;
    LayerCostModel _layerCostModel; // cost model for greedy strategy selection
    mlir::FuncOp _func;
    Logger _log;
};

//
// ConvolutionStrategy
//
class ConvolutionStrategy : public BaseLayerStrategy {
public:
    ConvolutionStrategy(mlir::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }

    bool doesLayerFitIntoCMX(mlir::Operation* op, StringRef strategy) const override final;
    bool isOperationSplitOverHeightCompatible(mlir::Operation* op) const override final;
};

//
// DepthConvolutionStrategy
//
class DepthConvolutionStrategy : public BaseLayerStrategy {
public:
    DepthConvolutionStrategy(mlir::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }

    bool doesLayerFitIntoCMX(mlir::Operation* op, StringRef strategy) const override final;
    bool isOperationSplitOverHeightCompatible(mlir::Operation* op) const override final;
};

//
// MaxPoolStrategy
//
class MaxPoolStrategy : public BaseLayerStrategy {
public:
    MaxPoolStrategy(mlir::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }

    bool doesLayerFitIntoCMX(mlir::Operation* op, StringRef strategy) const override final;
    bool isOperationSplitOverHeightCompatible(mlir::Operation* op) const override final;
};

//
// EltwiseStrategy
//
class EltwiseStrategy : public BaseLayerStrategy {
public:
    EltwiseStrategy(mlir::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }

    bool doesLayerFitIntoCMX(mlir::Operation* op, StringRef strategy) const override final;
};

//
// StrategyManager
//

// Higher level strategy manager class
// Its current purpose is to globally assign strategies
// In future it may have methods for finding sub-graphs
// and other strategy related utilites
class StrategyManager final {
public:
    explicit StrategyManager(mlir::FuncOp func, Logger log);

public:
    void assignMultiClusterStrategy();

private:
    void setLayerStrategy(const llvm::StringRef strategy, mlir::Operation* origOp) const;

    mlir::FuncOp _func;
    Logger _log;
    ConvolutionStrategy _convolutionStrategy;
    DepthConvolutionStrategy _depthConvolutionStrategy;
    MaxPoolStrategy _maxPoolStrategy;
    EltwiseStrategy _eltwiseStrategy;
};

// The efficiency calculation that is being performed here can be described as follows.
// A ratio of the real output tensor volume to the actual computation that occurs on the
// hardware for each MPE Mode 4x4x16 and 16x1x16 is computed and the maximum is selected.
template <class ConcreteOp>
double LayerCostModel::computeSplitEfficiency(ConcreteOp op, StringRef strategy) const {
    const auto perClusterShape = getLargestClusterOutputShape(op, strategy, _numClusters);
    const auto perClusterOutputTensorVolume =
            perClusterShape[Dims4D::Act::H] * perClusterShape[Dims4D::Act::W] * perClusterShape[Dims4D::Act::C];

    return std::max(static_cast<double>(perClusterOutputTensorVolume) /
                            calculateMPEVolume(VPU::MPEMode::MATRIX, perClusterShape),
                    static_cast<double>(perClusterOutputTensorVolume) /
                            calculateMPEVolume(VPU::MPEMode::VECTOR, perClusterShape));
}

template <class ConcreteOp>
StringRef BaseLayerStrategy::getOptimalLayerStrategy(ConcreteOp op, bool isTimeCost = true) const {
    double splitOverHeightCost = 0.0;
    double splitOverKernelCost = 0.0;
    const auto arch = VPU::getArch(op.getOperation());
    const auto isChannelMajor = (DimsOrder::fromValue(op.input()) == DimsOrder::NCHW) &&
                                VPU::NCEInvariant::isChannelMajorCompatible(
                                        arch, op.input().getType().template cast<vpux::NDTypeInterface>());

    if (isOperationSplitOverHeightCompatible(op) &&
        (doesLayerFitIntoCMX(op, splitOverHeightOverlapped) || doesLayerFitIntoCMX(op, splitOverHeight))) {
        splitOverHeightCost = _layerCostModel.getLayerCost(op, splitOverHeight, isTimeCost);
    }

    if (isOperationSplitOverKernelCompatible(op) && doesLayerFitIntoCMX(op, splitOverKernel)) {
        splitOverKernelCost = _layerCostModel.getLayerCost(op, splitOverKernel, isTimeCost);
    }

    // Compute ammount of clusters so that SOK is compatible
    const auto outputChannels = op.output().getType().template cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C];
    const auto sokOptimalClusters =
            getNumberOfClustersForSOKToAvoidAlignment(outputChannels, _numClusters);
    const auto optimalHeightTiling = [&](void) {
        return isChannelMajor ? splitOverHeightOverlapped : splitOverHeight;
    };
    
    if (sokOptimalClusters == _numClusters) {
        if (splitOverHeightCost >= splitOverKernelCost) {
            return optimalHeightTiling();
        }
        return splitOverKernel;
    } else {
        // SOK uses less clusters, but it would be still better than if
        // SOH is incompatible.
        if (splitOverHeightCost > 0) {
            return optimalHeightTiling();
        }
        return splitOverKernel;
    }
}

}  // namespace VPU
}  // namespace vpux
