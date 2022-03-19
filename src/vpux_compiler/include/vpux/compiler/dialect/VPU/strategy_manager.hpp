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

    virtual bool doesLayerFitIntoCMX(mlir::Operation* op, StringRef strategy) const = 0;
    bool isOperationSplitOverHeightCompatible(mlir::Operation* op) const;

protected:
    int64_t _numClusters;
    int64_t _numDPUs;
    int64_t _minimumOutputHeightForSOH;
    const size_t _numChannelAlignment = 16;
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
};

//
// DepthConvolutionStrategy
//
class DepthConvolutionStrategy : public BaseLayerStrategy {
public:
    DepthConvolutionStrategy(mlir::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }
    bool doesLayerFitIntoCMX(mlir::Operation* op, StringRef strategy) const override final;
};

//
// MaxPoolStrategy
//
class MaxPoolStrategy : public BaseLayerStrategy {
public:
    MaxPoolStrategy(mlir::FuncOp func, Logger log): BaseLayerStrategy(func, log) {
    }

    bool doesLayerFitIntoCMX(mlir::Operation* op, StringRef strategy) const override final;
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
// Subgraph optimizer
//
class SubgraphOptimizer final {
public:
    SubgraphOptimizer(mlir::FuncOp func, Logger log);
    
    void verifySpillStrategies(bool lockClusteringStrategy)

    bool addSpillsAtStrategyTransition()

    void rollbackOrSpill();

private:
    bool isSpillingFromStrategy(mlir::Operator* op);
    bool isKCompatible(mlir::Operator* op);
    mlir::FuncOp _func;
    Logger _log;
    double COST_MAX = numeric_limits<double>::infinity();
    // format: {opname: (parentSpilling, spilling)}
    std::unordered_map<std::string, std::pair<bool, bool>> _spillingInfo;
    // format: {opname: {strategy, doesSkip}}
    std::unordered_map<std::string, std::unordered_map<StringLiteral, bool>> _strategySkip;
    // incompatible strategies need spilling to connect
    std::vector<std::pair<std::string, std::string>> incompatibleStrategies =
    {
        {"SplitOverHOverlapped", "Clustering"},
        {"SplitOverHOverlapped", "SplitOverK"},
        {"SplitOverH", "Clustering"},
        {"SplitOverH", "SplitOverK"},
        {"SplitOverK", "SplitOverH"},
        {"SplitOverK", "HKSwitch"},
        {"Clustering", "SplitOverH"},
        {"Clustering", "HKSwitch"},
        {"HKSwitch", "SplitOverH"},
        {"HKSwitch", "HKSwitch"}
    };
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
    void optimizeMulticlusterStrategy();

private:
    void setLayerStrategy(const llvm::StringRef strategy, mlir::Operation* origOp) const;

    mlir::FuncOp _func;
    Logger _log;
    ConvolutionStrategy _convolutionStrategy;
    DepthConvolutionStrategy _depthConvolutionStrategy;
    MaxPoolStrategy _maxPoolStrategy;
    EltwiseStrategy _eltwiseStrategy;
    SubgraphOptimizer _optimizer;
};

}  // namespace VPU
}  // namespace vpux
