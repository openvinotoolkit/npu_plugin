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
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils.hpp"
#include "vpux/utils/core/checked_cast.hpp"
namespace vpux {

constexpr llvm::StringLiteral multiClusterStrategyAttrName = "multiClusterStrategy";

//
// StrategyManager
//

class StrategyManager final {
public:
    explicit StrategyManager(mlir::FuncOp func, size_t numClusters, Logger log);

public:
    void computeOptimalMultiClusterStrategy();

private:
    template <class ConcreteOp>
    bool isOperationSplitOverHeightCompatible(ConcreteOp op);
    template <class ConcreteOp>
    bool isOperationSplitOverKernelCompatible(ConcreteOp op);
    template <class ConcreteOp>
    void assignMultiClusterStrategyForEltwise(ConcreteOp& op);
    void assignMultiClusterStrategy(mlir::Operation* op);
    double calculateSplitOverHeightEfficency(mlir::Operation* op);
    double calculateSplitOverKernelEfficency(mlir::Operation* op);
    std::map<int64_t, std::map<int64_t, double>> channelMajorEfficiencyTable();
    std::map<int64_t, std::map<int64_t, double>> depthwiseEfficiencyTable();

    const long int _minimumHeightForSOH = 20;
    const long int _minimumOutputChannelsPerCluster = 16;
    // llvm::DenseMap<mlir::Operation*, double> _splitOverHeightEfficencies;
    // llvm::DenseMap<mlir::Operation*, double> _splitOverKernelEfficencies;
    std::map<mlir::Operation*, double> _splitOverHeightEfficencies;
    std::map<mlir::Operation*, double> _splitOverKernelEfficencies;
    Logger _log;
    long int _numClusters;
    size_t _numDPUPerCluster = 5;
    size_t _numDPU;
    size_t _numChannelAlignment = 16;
    mlir::FuncOp _func;
};

template <class ConcreteOp>
bool StrategyManager::isOperationSplitOverHeightCompatible(ConcreteOp op) {
    const auto outputShape = getShape(op.output());
    const auto OH = outputShape[Dims4D::Act::H];
    return OH >= _minimumHeightForSOH;
}

template <class ConcreteOp>
bool StrategyManager::isOperationSplitOverKernelCompatible(ConcreteOp op) {
    const auto outputShape = getShape(op.output());
    const auto OC = outputShape[Dims4D::Act::C];
    return OC >= _minimumOutputChannelsPerCluster * _numClusters;
}

template <class ConcreteOp>
void StrategyManager::assignMultiClusterStrategyForEltwise(ConcreteOp& op) {
    // If operation is not SOH compatible, then it has to be Clustering
    if (isOperationSplitOverHeightCompatible<ConcreteOp>(op)) {
        op->setAttr(multiClusterStrategyAttrName, mlir::StringAttr::get(op->getContext(), "SplitOverH"));
        _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'", op->getAttr(multiClusterStrategyAttrName),
                   op->getName());
    } else {
        op->setAttr(multiClusterStrategyAttrName, mlir::StringAttr::get(op->getContext(), "Clustering"));
        _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'", op->getAttr(multiClusterStrategyAttrName),
                   op->getName());
    }
};

}  // namespace vpux
