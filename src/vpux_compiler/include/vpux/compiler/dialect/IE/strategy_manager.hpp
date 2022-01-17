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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/utils/core/checked_cast.hpp"
namespace vpux {

//
// StrategyManager
//

class StrategyManager final {
public:
    explicit StrategyManager(mlir::FuncOp func, Logger log);

public:
    void computeOptimalMultiClusterStrategy();

private:
    template <class ConcreteOp>
    bool isOperationSplitOverHeightCompatible(ConcreteOp op);
    size_t calculateSplitOverHeightEfficency(mlir::Operation* op);
    Logger _log;
    mlir::FuncOp _func;
};

template <class ConcreteOp>
bool StrategyManager::isOperationSplitOverHeightCompatible(ConcreteOp op) {
    const auto outputShape = getShape(op.output());
    const auto OH = outputShape[Dims4D::Act::H];
    return OH >= 20;
}

}  // namespace vpux
