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
    bool isOperationSplitOverKernelCompatible(mlir::Operation* op) const;
    bool isOperationMultiClusterCompatible(mlir::Operation* op) const;

    template <class ConcreteOp>
    StringRef getOptimalLayerStrategy(ConcreteOp op) const;

protected:
    virtual bool doesLayerFitIntoCMX(mlir::Operation* op, StringRef strategy) const = 0;

    template <class ConcreteOp>
    double computeSplitEfficiency(ConcreteOp op, StringRef strategy) const;
    template <class ConcreteOp>
    double calculateMPEVolume(ConcreteOp op, VPU::MPEMode mpeMode, StringRef strategy) const;

protected:
    int64_t _numClusters;
    int64_t _numDPUs;
    int64_t _minimumOutputHeightForSOH;
    const int64_t _numChannelAlignment = 16;
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

// The function computes the actual per cluster output tensor volume (i.e. computation that is performed)
// given the stratey and the MPE mode
template <class ConcreteOp>
double BaseLayerStrategy::calculateMPEVolume(ConcreteOp op, VPU::MPEMode mpeMode, StringRef strategy) const {
    mlir::ArrayAttr activationAlignment = nullptr;
    double mpeHeight = 16;
    double mpeWidth = 1;
    if (strategy == splitOverKernel) {
        const auto activationTensorNumTiles = getIntArrayAttr(
                op.getContext(), getActivationTensorNumTiles(op.getOperation(), _numClusters, strategy));
        activationAlignment = getIntArrayAttr(
                op.getContext(),
                getActivationTensorAlignment(op.getOperation(), strategy, false, activationTensorNumTiles));
    }

    const auto outputTensorDistributionMode = getOutputTensorDistributionMode(strategy);
    const auto outputTensorNumTiles =
            getIntArrayAttr(op->getContext(), getOutputTensorNumTiles(op.getOperation(), _numClusters, strategy));
    const auto distributedOutputTensorType = createDistributedTensorType(
            op, op.output(), outputTensorDistributionMode, outputTensorNumTiles, activationAlignment, strategy);

    auto perClusterShape = distributedOutputTensorType.getLargestCompactShape();
    double perClusterOutputWidth = perClusterShape[Dims4D::Act::W];
    double perClusterOutputHeight = perClusterShape[Dims4D::Act::H];
    double perClusterOutputChannels = perClusterShape[Dims4D::Act::C];

    if (mpeMode == VPU::MPEMode::VECTOR) {
        mpeHeight = 16;
        mpeWidth = 1;
    } else if (mpeMode == VPU::MPEMode::MATRIX) {
        mpeHeight = 4;
        mpeWidth = 4;
    } else {
        VPUX_THROW("Unsupported MPE mode {0}", mpeMode);
    }

    return _numDPUs * std::ceil((mpeHeight * std::ceil(perClusterOutputHeight / mpeHeight) * mpeWidth *
                                 std::ceil(perClusterOutputWidth / mpeWidth) * _numChannelAlignment *
                                 std::ceil(perClusterOutputChannels / _numChannelAlignment)) /
                                _numDPUs);
}

// The efficiency calculation that is being performed here can be described as follows.
// A ratio of the real output tensor volume to the actual computation that occurs on the
// hardware for each MPE Mode 4x4x16 and 16x1x16 is computed and the maximum is selected.
template <class ConcreteOp>
double BaseLayerStrategy::computeSplitEfficiency(ConcreteOp op, StringRef strategy) const {
    double perClusterOutputTensorVolume = 0;
    mlir::ArrayAttr activationAlignment = nullptr;

    const auto outputTensorDistributionMode = getOutputTensorDistributionMode(strategy);
    const auto outputTensorNumTiles =
            getIntArrayAttr(op->getContext(), getOutputTensorNumTiles(op.getOperation(), _numClusters, strategy));
    if (strategy == splitOverKernel) {
        const auto activationTensorNumTiles = getIntArrayAttr(
                op.getContext(), getActivationTensorNumTiles(op.getOperation(), _numClusters, strategy));
        activationAlignment = getIntArrayAttr(
                op.getContext(),
                getActivationTensorAlignment(op.getOperation(), strategy, false, activationTensorNumTiles));
    }
    const auto distributedOutputTensorType = createDistributedTensorType(
            op, op.output(), outputTensorDistributionMode, outputTensorNumTiles, activationAlignment, strategy);

    const auto perClusterShape = distributedOutputTensorType.getLargestCompactShape();
    perClusterOutputTensorVolume =
            perClusterShape[Dims4D::Act::H] * perClusterShape[Dims4D::Act::W] * perClusterShape[Dims4D::Act::C];

    return std::max(perClusterOutputTensorVolume / calculateMPEVolume(op, VPU::MPEMode::MATRIX, strategy),
                    perClusterOutputTensorVolume / calculateMPEVolume(op, VPU::MPEMode::VECTOR, strategy));
}

template <class ConcreteOp>
StringRef BaseLayerStrategy::getOptimalLayerStrategy(ConcreteOp op) const {
    double splitOverHeightEfficiency = 0.0;
    double splitOverKernelEfficiency = 0.0;
    bool isChannelMajor = false;

    if (isOperationSplitOverHeightCompatible(op) &&
        (doesLayerFitIntoCMX(op, splitOverHeightOverlapped) || doesLayerFitIntoCMX(op, splitOverHeight))) {
        splitOverHeightEfficiency = computeSplitEfficiency(op, splitOverHeight);
    }

    if (isOperationSplitOverKernelCompatible(op) && doesLayerFitIntoCMX(op, splitOverKernel)) {
        splitOverKernelEfficiency = computeSplitEfficiency(op, splitOverKernel);
    }

    if (splitOverHeightEfficiency >= splitOverKernelEfficiency) {
        return isChannelMajor ? splitOverHeightOverlapped : splitOverHeight;
    }
    return splitOverKernel;
}

}  // namespace VPU
}  // namespace vpux
