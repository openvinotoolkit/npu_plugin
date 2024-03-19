//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// EliminateConcat
//

class EliminateConcat final : public mlir::OpRewritePattern<VPU::ConcatOp> {
public:
    EliminateConcat(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::ConcatOp>(ctx), _log(log) {
        setDebugName("EliminateConcat");
    }

    mlir::LogicalResult matchAndRewrite(VPU::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EliminateConcat::matchAndRewrite(VPU::ConcatOp origOp, mlir::PatternRewriter& rewriter) const {
    if (!origOp.getStaticOffsets().has_value()) {
        return mlir::failure();
    }

    const auto concatOffsets = parseIntArrayOfArrayAttr<int64_t>(origOp.getStaticOffsets().value());
    DenseMap<VPU::SliceOp, std::pair<SmallVector<int64_t>, mlir::Value>> newSliceOffsetsInputMap;

    const auto allUsersSliceSubTensors = llvm::all_of(origOp->getUsers(), [&](auto userOp) {
        auto maybeSliceOp = mlir::dyn_cast_or_null<VPU::SliceOp>(userOp);
        if (maybeSliceOp == nullptr) {
            return false;
        }

        auto sliceOffset = parseIntArrayAttr<int64_t>(maybeSliceOp.getStaticOffsets());
        const auto sliceOutShape = getShape(maybeSliceOp.getResult()).raw();

        for (const auto& p : zip(origOp.getInputs(), concatOffsets)) {
            const auto concatInput = std::get<0>(p);
            const auto concatInputShape = getShape(concatInput).raw();
            const auto concatOffset = std::get<1>(p);

            if (auto inputOp = concatInput.getDefiningOp()) {
                if (!inputOp->hasOneUse()) {
                    continue;
                }
            }

            const auto isSubTensor = [&]() -> bool {
                for (const auto dim : irange(sliceOutShape.size())) {
                    if ((sliceOffset[dim] < concatOffset[dim]) ||
                        (concatOffset[dim] + concatInputShape[dim] < sliceOffset[dim] + sliceOutShape[dim])) {
                        return false;
                    }
                }
                return true;
            };

            if (!isSubTensor()) {
                continue;
            }

            for (const auto dim : irange(sliceOffset.size())) {
                sliceOffset[dim] -= concatOffset[dim];
            }

            newSliceOffsetsInputMap[maybeSliceOp] = std::pair{sliceOffset, concatInput};
            return true;
        }

        return false;
    });

    if (!allUsersSliceSubTensors) {
        return mlir::failure();
    }

    _log.trace("The Concat at {0} is eliminated", origOp.getLoc());

    for (const auto& keyValue : newSliceOffsetsInputMap) {
        auto origSlice = keyValue.first;
        const auto sliceOffset = keyValue.second.first;
        const auto sliceInput = keyValue.second.second;

        rewriter.setInsertionPoint(origSlice);
        rewriter.replaceOpWithNewOp<VPU::SliceOp>(origSlice, origSlice.getResult().getType(), sliceInput,
                                                  getIntArrayAttr(getContext(), sliceOffset),
                                                  origSlice.getStaticSizes());
    }

    return mlir::success();
}

//
// OptimizeConcatPass
//

class OptimizeConcatPass final : public VPU::OptimizeConcatBase<OptimizeConcatPass> {
public:
    explicit OptimizeConcatPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeConcatPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<EliminateConcat>(&ctx, _log);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeConcatPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createOptimizeConcatPass(Logger log) {
    return std::make_unique<OptimizeConcatPass>(log);
}
