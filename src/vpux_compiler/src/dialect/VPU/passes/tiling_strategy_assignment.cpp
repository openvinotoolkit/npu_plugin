//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

//
// TilingStrategyAssignment
//
// There are three modes of tiling strategies as defined in vpux::TilingMode:
// 1. ISOLATED tiling: Split operations with smallest tiling number to make them fit into CMX
// 2. PIPELINING tiling: Overlap the DPU time of earlier sub-tile with the DMA time of the later ones
//              Two possible scenarios where the PIPELINING could be triggered:
//              a). When ISOLATED tiling is required, the tiling number will be increased to satisfy PIPELINING
//              b). When the constant weights of an operation is larger than the threshold
//                  tiling number will be increased to satisfy PIPELINING
//                  even though the operation doesn't require ISOLATED tiling originally
//              A precondition is that a feasible tiling strategy must exist to make PIPELINING work
//              Otherwise it will fallback to ISOLATED tiling or non-tiling
// 3. PREFETCHING tiling: Overlap the DPU time of parent operation with the DMA time the child
//

//
// TilingStrategyAssignmentPass
//
class TilingStrategyAssignmentPass final : public VPU::TilingStrategyAssignmentBase<TilingStrategyAssignmentPass> {
public:
    explicit TilingStrategyAssignmentPass(bool enablePrefetchTiling, Logger log)
            : _enablePrefetchTiling(enablePrefetchTiling) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    void assignStrategy(VPU::TilingBuilderOpInterface origOp);

    bool _enablePrefetchTiling = true;
};

mlir::LogicalResult TilingStrategyAssignmentPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (tilingMode.hasValue()) {
        _log.trace("Overloading C++ createTilingStrategyAssignmentPass argument by MLIR variable");
        _enablePrefetchTiling = tilingMode.getValue() == "PREFETCH";
    }
    return mlir::success();
}

void TilingStrategyAssignmentPass::assignStrategy(VPU::TilingBuilderOpInterface origOp) {
    _log.trace("Assign: '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto op = origOp.getOperation();
    const auto tiles = getLayerTilingStrategy(origOp, _enablePrefetchTiling, _log);
    VPUX_THROW_WHEN(mlir::failed(tiles), "Invalid tiling strategy for {0}", origOp->getLoc());

    origOp->setAttr(tilingStrategy, getIntArrayAttr(op->getContext(), tiles.value()[0].axis));
}

void TilingStrategyAssignmentPass::safeRunOnFunc() {
    auto func = getOperation();

    const auto callback = [&](mlir::Operation* op) {
        auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
        if (tilingOp != nullptr && VPU::opNeedsTiling(op, _enablePrefetchTiling, _log)) {
            assignStrategy(tilingOp);
        }
    };

    func->walk(callback);
}
}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createTilingStrategyAssignmentPass(bool enablePrefetchTiling, Logger log) {
    return std::make_unique<TilingStrategyAssignmentPass>(enablePrefetchTiling, log);
}
