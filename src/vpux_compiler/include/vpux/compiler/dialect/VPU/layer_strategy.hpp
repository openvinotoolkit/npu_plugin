//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/checked_cast.hpp"

namespace vpux {
namespace VPU {

constexpr int64_t SINGLE_BATCH = 1;
constexpr size_t RANK_REQUIRED_FOR_TILING = 4;

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
    explicit BaseLayerStrategy(mlir::func::FuncOp func, Logger log);
    virtual ~BaseLayerStrategy() = default;

    virtual bool isOperationSplitOverHeightCompatible(VPU::ClusteredOpInterface nceOp,
                                                      ShapeRef customOutputShape = ShapeRef()) const;
    virtual bool isOperationSplitOverWidthCompatible(VPU::ClusteredOpInterface nceOp,
                                                     ShapeRef customOutputShape = ShapeRef()) const;
    virtual bool isOperationSplitOverKernelCompatible(VPU::ClusteredOpInterface nceOp,
                                                      ShapeRef customOutputShape = ShapeRef()) const;

    virtual SmallVector<VPU::DistributedTypeInterface> getDistributedTensorType(
            VPU::ClusteredOpInterface nceOp, VPU::MultiClusterStrategy strategy) const = 0;
    virtual bool doesLayerFitIntoCMX(VPU::ClusteredOpInterface nceOp, VPU::MultiClusterStrategy strategy,
                                     Byte reservedMem = Byte(0)) const;
    virtual bool doesLayerChangeOutputAlignmentFitIntoCMX(VPU::ClusteredOpInterface nceOp,
                                                          VPU::MultiClusterStrategy strategy,
                                                          VPU::DistributedTypeInterface newDistributedTensorType) const;

protected:
    int64_t _numClusters;
    int64_t _numDPUs;
    int64_t _minimumOutputHeightForSOH;
    int64_t _minimumOutputWidthForSOW;
    const int64_t _numChannelAlignment = 16;
    mlir::func::FuncOp _func;
    Logger _log;
};

//
// ConvolutionStrategy
//
class ConvolutionStrategy : public BaseLayerStrategy {
public:
    ConvolutionStrategy(mlir::func::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }

    SmallVector<VPU::DistributedTypeInterface> getDistributedTensorType(VPU::ClusteredOpInterface nceOp,
                                                                        VPU::MultiClusterStrategy strategy) const final;
    bool isOperationSplitOverHeightCompatible(VPU::ClusteredOpInterface nceOp,
                                              ShapeRef customOutputShape = ShapeRef()) const final;
};

//
// CompressConvolutionStrategy
//
class CompressConvolutionStrategy : public BaseLayerStrategy {
public:
    CompressConvolutionStrategy(mlir::func::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }

    SmallVector<VPU::DistributedTypeInterface> getDistributedTensorType(VPU::ClusteredOpInterface nceOp,
                                                                        VPU::MultiClusterStrategy strategy) const final;
    bool isOperationSplitOverHeightCompatible(VPU::ClusteredOpInterface nceOp,
                                              ShapeRef customOutputShape = ShapeRef()) const final;
};

//
// DepthConvolutionStrategy
//
class DepthConvolutionStrategy : public BaseLayerStrategy {
public:
    DepthConvolutionStrategy(mlir::func::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }

    SmallVector<VPU::DistributedTypeInterface> getDistributedTensorType(VPU::ClusteredOpInterface nceOp,
                                                                        VPU::MultiClusterStrategy strategy) const final;
    bool isOperationSplitOverHeightCompatible(VPU::ClusteredOpInterface nceOp,
                                              ShapeRef customOutputShape = ShapeRef()) const final;
};

//
// MaxPoolStrategy
//
class MaxPoolStrategy : public BaseLayerStrategy {
public:
    MaxPoolStrategy(mlir::func::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }

    SmallVector<VPU::DistributedTypeInterface> getDistributedTensorType(VPU::ClusteredOpInterface nceOp,
                                                                        VPU::MultiClusterStrategy strategy) const final;
    bool isOperationSplitOverHeightCompatible(VPU::ClusteredOpInterface nceOp,
                                              ShapeRef customOutputShape = ShapeRef()) const final;
};

//
// AveragePoolStrategy
//
class AveragePoolStrategy : public BaseLayerStrategy {
public:
    AveragePoolStrategy(mlir::func::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }

    SmallVector<VPU::DistributedTypeInterface> getDistributedTensorType(VPU::ClusteredOpInterface nceOp,
                                                                        VPU::MultiClusterStrategy strategy) const final;
    bool isOperationSplitOverHeightCompatible(VPU::ClusteredOpInterface nceOp,
                                              ShapeRef customOutputShape = ShapeRef()) const final;
};

//
// EltwiseStrategy
//
class EltwiseStrategy : public BaseLayerStrategy {
public:
    EltwiseStrategy(mlir::func::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }

    SmallVector<VPU::DistributedTypeInterface> getDistributedTensorType(VPU::ClusteredOpInterface nceOp,
                                                                        VPU::MultiClusterStrategy strategy) const final;
};

//
// PermuteQuantizeStrategy
//
class PermuteQuantizeStrategy : public BaseLayerStrategy {
public:
    PermuteQuantizeStrategy(mlir::func::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }

    SmallVector<VPU::DistributedTypeInterface> getDistributedTensorType(VPU::ClusteredOpInterface nceOp,
                                                                        VPU::MultiClusterStrategy strategy) const final;
};

//
// InterpolateStrategy
//
class InterpolateStrategy : public BaseLayerStrategy {
public:
    InterpolateStrategy(mlir::func::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }

    SmallVector<VPU::DistributedTypeInterface> getDistributedTensorType(VPU::ClusteredOpInterface nceOp,
                                                                        VPU::MultiClusterStrategy strategy) const final;
    bool isOperationSplitOverHeightCompatible(VPU::ClusteredOpInterface nceOp,
                                              ShapeRef customOutputShape = ShapeRef()) const final;
};

//
// ConcatStrategy
//
class ConcatStrategy : public BaseLayerStrategy {
public:
    ConcatStrategy(mlir::func::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }

    SmallVector<VPU::DistributedTypeInterface> getDistributedTensorType(VPU::ClusteredOpInterface nceOp,
                                                                        VPU::MultiClusterStrategy strategy) const final;
    bool isOperationSplitOverHeightCompatible(VPU::ClusteredOpInterface nceOp,
                                              ShapeRef customOutputShape = ShapeRef()) const final;
    bool isOperationSplitOverKernelCompatible(VPU::ClusteredOpInterface nceOp,
                                              ShapeRef customOutputShape = ShapeRef()) const final;
};

//
// SWGeneralStrategy
//
class SWGeneralStrategy : public BaseLayerStrategy {
public:
    SWGeneralStrategy(mlir::func::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }

    virtual bool isOperationSplitOverHeightCompatible(VPU::ClusteredOpInterface clusteredOp,
                                                      ShapeRef customOutputShape = ShapeRef()) const final;
    virtual bool isOperationSplitOverWidthCompatible(VPU::ClusteredOpInterface clusteredOp,
                                                     ShapeRef customOutputShape = ShapeRef()) const final;
    virtual bool isOperationSplitOverKernelCompatible(VPU::ClusteredOpInterface clusteredOp,
                                                      ShapeRef customOutputShape = ShapeRef()) const final;

    SmallVector<VPU::DistributedTypeInterface> getDistributedTensorType(VPU::ClusteredOpInterface nceOp,
                                                                        VPU::MultiClusterStrategy strategy) const final;

private:
    bool checkMCRestrictions(VPU::ClusteredOpInterface clusteredOp) const;
};

//
// SWInterpolateStrategy
//
class SWInterpolateStrategy : public BaseLayerStrategy {
public:
    SWInterpolateStrategy(mlir::func::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }

    SmallVector<VPU::DistributedTypeInterface> getDistributedTensorType(VPU::ClusteredOpInterface nceOp,
                                                                        VPU::MultiClusterStrategy strategy) const final;
};

//
// SWMultiplyStrategy
//
class SWMultiplyStrategy : public BaseLayerStrategy {
public:
    SWMultiplyStrategy(mlir::func::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }

    SmallVector<VPU::DistributedTypeInterface> getDistributedTensorType(VPU::ClusteredOpInterface nceOp,
                                                                        VPU::MultiClusterStrategy strategy) const final;
};

//
// SWTopKStrategy
//
class SWTopKStrategy : public BaseLayerStrategy {
public:
    SWTopKStrategy(mlir::func::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }

    SmallVector<VPU::DistributedTypeInterface> getDistributedTensorType(VPU::ClusteredOpInterface swOp,
                                                                        VPU::MultiClusterStrategy strategy) const final;
};

}  // namespace VPU
}  // namespace vpux
