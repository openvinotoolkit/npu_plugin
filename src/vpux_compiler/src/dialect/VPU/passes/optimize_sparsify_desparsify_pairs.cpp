//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// OptimizeSparsifyDesparsifyPairsPass
//

class OptimizeSparsifyDesparsifyPairsPass final :
        public VPU::OptimizeSparsifyDesparsifyPairsBase<OptimizeSparsifyDesparsifyPairsPass> {
public:
    explicit OptimizeSparsifyDesparsifyPairsPass(VPU::SparsityProfileCreateFunc sparsityProfileCreateCb, Logger log)
            : _sparsityProfileCreateCb(std::move(sparsityProfileCreateCb)) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

    VPU::ActivationSparsityProfile _sparsityProfile{VPU::ActivationSparsityProfile::S0};
    VPU::SparsityProfileCreateFunc _sparsityProfileCreateCb;
};

mlir::LogicalResult OptimizeSparsifyDesparsifyPairsPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    const auto parsedSparsityProfile = _sparsityProfileCreateCb(sparsityProfile.getValue());
    if (!parsedSparsityProfile.hasValue()) {
        return mlir::failure();
    }
    _sparsityProfile = parsedSparsityProfile.getValue();
    return mlir::success();
}

//
// RemoveConcatWrappers
//

class RemoveConcatWrappers final : public mlir::OpRewritePattern<VPU::ConcatOp> {
public:
    RemoveConcatWrappers(mlir::MLIRContext* ctx, Logger log, VPU::ActivationSparsityProfile sparsityProfile)
            : mlir::OpRewritePattern<VPU::ConcatOp>(ctx), _log(log), _sparsityProfile(sparsityProfile) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::ConcatOp quantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    VPU::ActivationSparsityProfile _sparsityProfile;
};

// Legend:
// === sparse path
// --- dense path
// Convert VPU.Concat ops wrapped by Desparsify/Sparsify
// => Desparsify \                / Sparsify => SparseConsumer
// => Desparsify |-> VPU.Concat ->| Sparsify => SparseConsumer
// => Desparsify /                \ DenseConsumer(if S1 profile)
// To
// ==\                 /=> SparseConsumer
// ==| => VPU.Concat =>|=>  SparseConsumer
// ==/                 \=> Desparsify => DenseConsumer
mlir::LogicalResult RemoveConcatWrappers::matchAndRewrite(VPU::ConcatOp concatOp,
                                                          mlir::PatternRewriter& rewriter) const {
    // Allow mixed consumers depends on profile. TODO: E#53581
    const bool allowMixedConsumers = _sparsityProfile == VPU::ActivationSparsityProfile::S1;

    mlir::DenseSet<mlir::Operation*> consumerSparsifyOps;
    size_t totalConsumers = 0;
    for (auto user : concatOp.output().getUsers()) {
        ++totalConsumers;
        if (mlir::isa<VPU::SparsifyOp>(user)) {
            consumerSparsifyOps.insert(user);
        }
    }

    if (consumerSparsifyOps.empty()) {
        return mlir::failure();
    }

    const auto canBeFullyConverted = consumerSparsifyOps.size() == totalConsumers;
    if (!allowMixedConsumers && !canBeFullyConverted) {
        _log.trace("Couldnt remove sparsity wrappers around '{0}', because mixed consumers are not allowed.", concatOp);
        return mlir::failure();
    }

    SmallVector<mlir::Value> newConcatInputs;
    SmallVector<VPU::DesparsifyOp> removableDesparsifyOps;
    for (mlir::Value input : concatOp.inputs()) {
        if (auto producingDesparsifyOp = input.getDefiningOp<VPU::DesparsifyOp>()) {
            newConcatInputs.push_back(producingDesparsifyOp.input());
            removableDesparsifyOps.push_back(producingDesparsifyOp);
        }
    }

    if (newConcatInputs.size() != concatOp.inputs().size()) {
        return mlir::failure();
    }

    _log.trace("Removed wrappers around '{0}'", concatOp);
    const auto newOutputType = (*consumerSparsifyOps.begin())->getResult(0).getType();
    auto newConcatOp = rewriter.create<VPU::ConcatOp>(concatOp->getLoc(), newOutputType, newConcatInputs,
                                                      concatOp.static_offsetsAttr());

    if (!canBeFullyConverted) {
        auto desparsifyOp = rewriter.create<VPU::DesparsifyOp>(concatOp->getLoc(), newConcatOp.output());
        concatOp.output().replaceUsesWithIf(desparsifyOp.output(), [&](mlir::OpOperand& opOperand) {
            return !consumerSparsifyOps.contains(opOperand.getOwner());
        });
    }

    for (auto consumingSparsifyOp : consumerSparsifyOps) {
        consumingSparsifyOp->getResult(0).replaceAllUsesWith(newConcatOp);
        rewriter.eraseOp(consumingSparsifyOp);
    }

    for (auto producingDesparsify : removableDesparsifyOps) {
        if (producingDesparsify.output().use_empty()) {
            rewriter.eraseOp(producingDesparsify);
        }
    }

    return mlir::success();
}

//
// EliminateDesparsifySparsifyPairs
//

class EliminateDesparsifySparsifyPairs final : public mlir::OpRewritePattern<VPU::DesparsifyOp> {
public:
    EliminateDesparsifySparsifyPairs(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::DesparsifyOp>(ctx), _log(log) {
        setDebugName("EliminateDesparsifySparsifyPairs");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::DesparsifyOp quantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Eliminate Desparsify->Sparsify sequences. In most general case looks like
//                /-> Sparsify => SparseConsumer
// => Desparsify -|-> Sparsify => SparseConsumer
//                \-> DenseConsumer
// To
// => Desparsify -> DenseConsumer
// ||=> SparseConsumer
// ||=> SparseConsumer
// If Desparsify has no dense consumers(most common case) sequence will be completely eliminated
mlir::LogicalResult EliminateDesparsifySparsifyPairs::matchAndRewrite(VPU::DesparsifyOp desparsifyOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    size_t totalUses = 0;
    mlir::DenseSet<mlir::Operation*> removableOps;
    for (const auto user : desparsifyOp->getUsers()) {
        ++totalUses;
        if (mlir::isa<VPU::SparsifyOp>(user)) {
            removableOps.insert(user);
        }
    }
    if (removableOps.empty()) {
        return mlir::failure();
    }
    auto sparseInput = desparsifyOp.input();
    for (auto sparsifyOp : removableOps) {
        sparsifyOp->getResult(0).replaceAllUsesWith(sparseInput);
        rewriter.eraseOp(sparsifyOp);
    }
    if (removableOps.size() == totalUses) {
        _log.trace("Pattern Desparsify->Sparsify completely erased with root at '{0}'", desparsifyOp);
        rewriter.eraseOp(desparsifyOp);
    }
    return mlir::success();
}

//
// safeRunOnFunc
//

void OptimizeSparsifyDesparsifyPairsPass::safeRunOnFunc() {
    using namespace VPU;
    using namespace VPU::NCESparsity;

    auto func = getFunction();
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<EliminateDesparsifySparsifyPairs>(&ctx, _log);
    patterns.add<RemoveConcatWrappers>(&ctx, _log, _sparsityProfile);
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeSparsifyDesparsifyPairsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createOptimizeSparsifyDesparsifyPairsPass(
        SparsityProfileCreateFunc sparsityProfileCreateCb, Logger log) {
    return std::make_unique<OptimizeSparsifyDesparsifyPairsPass>(sparsityProfileCreateCb, log);
}
