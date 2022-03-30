//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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
// cost model E#26043.
class BaseLayerStrategy {
public:
    explicit BaseLayerStrategy(mlir::FuncOp func, Logger log);
    virtual ~BaseLayerStrategy() = default;

    virtual bool isOperationSplitOverHeightCompatible(VPU::NCEOpInterface nceOp) const;
    virtual bool isOperationSplitOverKernelCompatible(VPU::NCEOpInterface nceOp) const;
    virtual bool isOperationMultiClusterCompatible(VPU::NCEOpInterface nceOp) const;

    VPU::MultiClusterStrategy getOptimalLayerStrategy(VPU::NCEOpInterface nceOp) const;

protected:
    virtual bool doesLayerFitIntoCMX(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const = 0;

    double computeSplitEfficiency(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const;
    double calculateMPEVolume(VPU::MPEMode mpeMode, Shape shape) const;

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

    bool doesLayerFitIntoCMX(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const override final;
    bool isOperationSplitOverHeightCompatible(VPU::NCEOpInterface nceOp) const override final;
};

//
// DepthConvolutionStrategy
//
class DepthConvolutionStrategy : public BaseLayerStrategy {
public:
    DepthConvolutionStrategy(mlir::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }

    bool doesLayerFitIntoCMX(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const override final;
    bool isOperationSplitOverHeightCompatible(VPU::NCEOpInterface nceOp) const override final;
};

//
// MaxPoolStrategy
//
class MaxPoolStrategy : public BaseLayerStrategy {
public:
    MaxPoolStrategy(mlir::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }

    bool doesLayerFitIntoCMX(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const override final;
    bool isOperationSplitOverHeightCompatible(VPU::NCEOpInterface nceOp) const override final;
};

//
// EltwiseStrategy
//
class EltwiseStrategy : public BaseLayerStrategy {
public:
    EltwiseStrategy(mlir::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }

    bool doesLayerFitIntoCMX(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const override final;
};

//
// StrategyManager
//

// Higher level strategy manager class
// Its current purpose is to globally assign strategies
// In future it may have methods for finding sub-graphs
// and other strategy related utilities
class StrategyManager final {
public:
    explicit StrategyManager(mlir::FuncOp func, Logger log);

public:
    void assignMultiClusterStrategy();

private:
    void setLayerStrategy(VPU::MultiClusterStrategy strategy, VPU::NCEOpInterface nceOp);
    bool overrideStrategyForLayer(VPU::MultiClusterStrategy strategy, VPU::NCEOpInterface nceOp);

    mlir::FuncOp _func;
    Logger _log;
    ConvolutionStrategy _convolutionStrategy;
    DepthConvolutionStrategy _depthConvolutionStrategy;
    MaxPoolStrategy _maxPoolStrategy;
    EltwiseStrategy _eltwiseStrategy;
};

}  // namespace VPU
}  // namespace vpux
