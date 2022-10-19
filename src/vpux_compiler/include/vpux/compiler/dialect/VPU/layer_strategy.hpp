//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/checked_cast.hpp"

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
    using Ptr = std::shared_ptr<BaseLayerStrategy>;
    explicit BaseLayerStrategy(mlir::FuncOp func, Logger log);
    virtual ~BaseLayerStrategy() = default;

    virtual bool isOperationSplitOverHeightCompatible(VPU::NCEOpInterface nceOp) const;
    virtual bool isOperationSplitOverKernelCompatible(VPU::NCEOpInterface nceOp) const;
    virtual bool isOperationMultiClusterCompatible(VPU::NCEOpInterface nceOp) const;

    virtual SmallVector<VPU::DistributedTensorType> getDistributedTensorType(
            VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const = 0;
    virtual bool doesLayerFitIntoCMX(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const;
    virtual bool doesLayerChangeOutputAlignmentFitIntoCMX(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy,
                                                          VPU::DistributedTensorType newDistributedTensorType) const;

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

    SmallVector<VPU::DistributedTensorType> getDistributedTensorType(
            VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const override final;
    bool isOperationSplitOverHeightCompatible(VPU::NCEOpInterface nceOp) const override final;
};

//
// DepthConvolutionStrategy
//
class DepthConvolutionStrategy : public BaseLayerStrategy {
public:
    DepthConvolutionStrategy(mlir::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }

    SmallVector<VPU::DistributedTensorType> getDistributedTensorType(
            VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const override final;
    bool isOperationSplitOverHeightCompatible(VPU::NCEOpInterface nceOp) const override final;
};

//
// MaxPoolStrategy
//
class MaxPoolStrategy : public BaseLayerStrategy {
public:
    MaxPoolStrategy(mlir::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }

    SmallVector<VPU::DistributedTensorType> getDistributedTensorType(
            VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const override final;
    bool isOperationSplitOverHeightCompatible(VPU::NCEOpInterface nceOp) const override final;
};

//
// AveragePoolStrategy
//
class AveragePoolStrategy : public BaseLayerStrategy {
public:
    AveragePoolStrategy(mlir::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }

    SmallVector<VPU::DistributedTensorType> getDistributedTensorType(
            VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const override final;
    bool isOperationSplitOverHeightCompatible(VPU::NCEOpInterface nceOp) const override final;
};

//
// EltwiseStrategy
//
class EltwiseStrategy : public BaseLayerStrategy {
public:
    EltwiseStrategy(mlir::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }

    SmallVector<VPU::DistributedTensorType> getDistributedTensorType(
            VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const override final;
};

}  // namespace VPU
}  // namespace vpux
