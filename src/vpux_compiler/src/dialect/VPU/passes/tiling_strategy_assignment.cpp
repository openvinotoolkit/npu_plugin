//
// Copyright (C) 2023 Intel Corporation
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
    bool isLegalOp(mlir::Operation* op);
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

bool TilingStrategyAssignmentPass::isLegalOp(mlir::Operation* op) {
    if (mlir::isa<VPU::SliceOp, VPU::ConcatOp, VPU::NCEClusterTilingOp>(op) ||
        op->getParentOfType<VPU::NCEClusterTilingOp>() || op->hasAttr(tilingStrategy)) {
        return true;
    }
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);
    if (!mlir::isa<VPU::NCEOpInterface>(op) && !VPU::archSupportsSwLayerTiling(arch)) {
        return true;
    }
    if (op->hasAttr(tilingStrategy)) {
        return true;
    }
    if (auto iface = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op)) {
        _log.trace("Check: '{0}' at '{1}'", op->getName(), op->getLoc());
        const auto resShape = getShape(op->getResult(0));
        if (!iface.isSupportedTiling({TileInfo(resShape)}, TilingMode::ISOLATED, _log.nest())) {
            _log.nest().trace("ISOLATED tiling or PIPELINING tiling required");
            return false;
        }
        if (_enablePrefetchTiling && mlir::isa<VPU::NCEOpInterface>(op)) {
            if (VPU::prefetchTilingConditionSatisfied(op, _log.nest())) {
                _log.nest().trace("PREFETCHING tiling required");
                return false;
            }
            if (VPU::largeConstPipelineConditionSatisfied(op, _log.nest())) {
                _log.nest().trace("PIPELINING tiling for large constant weights required");
                return false;
            }
        }
    }
    return true;
}

void TilingStrategyAssignmentPass::assignStrategy(VPU::TilingBuilderOpInterface origOp) {
    _log.trace("Assign: '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto op = origOp.getOperation();
    const auto tiles = getLayerTilingStrategy(origOp, _enablePrefetchTiling, _log);

    origOp->setAttr(tilingStrategy, getIntArrayAttr(op->getContext(), tiles[0].axis));
}

void TilingStrategyAssignmentPass::safeRunOnFunc() {
    auto func = getOperation();

    const auto callback = [&](mlir::Operation* op) {
        auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
        if (tilingOp != nullptr && !isLegalOp(op)) {
            assignStrategy(tilingOp);
        }
    };

    func->walk(callback);
}
}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createTilingStrategyAssignmentPass(bool enablePrefetchTiling, Logger log) {
    return std::make_unique<TilingStrategyAssignmentPass>(enablePrefetchTiling, log);
}
