//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// Common Utils
//

NDTypeInterface inferNewTypeWithMemPerm(NDTypeInterface oldType, mlir::AffineMap memPerm, const DimsOrder& dstOrder) {
    const auto oldMemShape = oldType.getMemShape();
    const auto newMemShape = applyPerm(oldMemShape, memPerm);
    const auto newShape = dstOrder.toLogicalOrder(newMemShape);
    return oldType.changeDimsOrder(dstOrder).changeShape(newShape);
}

IE::MemPermuteOp getSupportedInputMemPermute(mlir::Value input) {
    auto inputMemPermuteOp = input.getDefiningOp<IE::MemPermuteOp>();
    if (inputMemPermuteOp == nullptr || !inputMemPermuteOp->hasOneUse() ||
        getSupportedInputMemPermute(inputMemPermuteOp.input()) != nullptr) {
        return nullptr;
    }
    return inputMemPermuteOp;
}

IE::MemPermuteOp getSupportedOutputMemPermute(mlir::Value output) {
    if (!output.hasOneUse()) {
        return nullptr;
    }
    auto outputMemPermuteOp = mlir::dyn_cast_or_null<IE::MemPermuteOp>(*output.getUsers().begin());
    if (outputMemPermuteOp == nullptr || getSupportedOutputMemPermute(outputMemPermuteOp.output()) != nullptr) {
        return nullptr;
    }
    return outputMemPermuteOp;
}

//
// AdjustForEltwise
//
// This pattern tries to adjust the mempermutes around an eltwise to find the solution
// with least number of nontrivial permutes
class AdjustForEltwise final : public mlir::OpInterfaceRewritePattern<IE::LayerOpInterface> {
public:
    AdjustForEltwise(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<IE::LayerOpInterface>(ctx), _log(log) {
        setDebugName("AdjustForEltwise");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LayerOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Calculate the number of non-trivial permutes around the eltwise if inserting
// mempermutes with given permutation for all inputs of the layerOp
int64_t calcNumNonTrivialPermutesAroundEltwiseWithMemPerm(IE::LayerOpInterface layerOp, mlir::AffineMap newMemPerm) {
    auto ctx = layerOp.getContext();
    auto idMap = mlir::AffineMap::getMultiDimIdentityMap(newMemPerm.getNumDims(), ctx);
    int64_t totalNumNonTrivialPermutes = 0;
    // calculate number of mempermutes on input side
    for (auto input : layerOp->getOperands()) {
        auto inputMemPermuteOp = getSupportedInputMemPermute(input);
        auto permuteInput = (inputMemPermuteOp == nullptr) ? input : inputMemPermuteOp.input();
        const auto inMemPerm = (inputMemPermuteOp == nullptr) ? idMap : inputMemPermuteOp.mem_perm();
        const auto inMemShape = getMemShape(permuteInput);
        const auto newInMemPerm = newMemPerm.compose(inMemPerm);
        if (!isTrivialPermute(inMemShape, newInMemPerm)) {
            totalNumNonTrivialPermutes++;
        }
    }
    // calculate number of mempermutes on output side
    auto outMemPerm = idMap;
    if (layerOp->hasOneUse()) {
        auto outputMemPermuteOp = mlir::dyn_cast_or_null<IE::MemPermuteOp>(*layerOp->getUsers().begin());
        if (outputMemPermuteOp != nullptr) {
            outMemPerm = outputMemPermuteOp.mem_perm();
        }
    }
    const auto newMemShape = applyPerm(getMemShape(layerOp->getResult(0)), newMemPerm);
    const auto newOutMemPerm = outMemPerm.compose(mlir::inversePermutation(newMemPerm));
    if (!isTrivialPermute(newMemShape, newOutMemPerm)) {
        totalNumNonTrivialPermutes++;
    }
    return totalNumNonTrivialPermutes;
}

mlir::LogicalResult AdjustForEltwise::matchAndRewrite(IE::LayerOpInterface origOp,
                                                      mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto ctx = getContext();

    if (!origOp->hasTrait<IE::EltwiseOp>()) {
        return matchFailed(rewriter, origOp, "LayerOp is not Eltwise");
    }

    const auto outType = origOp->getResult(0).getType().cast<NDTypeInterface>();
    const auto outOrder = outType.getDimsOrder();
    const auto rank = outType.getRank();
    const auto idMap = mlir::AffineMap::getMultiDimIdentityMap(checked_cast<unsigned>(rank), getContext());

    auto bestMemPerm = idMap;
    auto bestNumNonTrivialPermutes = calcNumNonTrivialPermutesAroundEltwiseWithMemPerm(origOp, bestMemPerm);
    const auto checkBetterMemPerm = [&](mlir::AffineMap newMemPerm, int64_t origNumMemPermutes) -> Optional<int64_t> {
        const auto numNonTrivialPermutes = calcNumNonTrivialPermutesAroundEltwiseWithMemPerm(origOp, newMemPerm);
        if (numNonTrivialPermutes >= origNumMemPermutes) {
            return None;
        }
        return numNonTrivialPermutes;
    };

    // try with input permutes
    for (auto input : origOp->getOperands()) {
        // input order should be same as output order
        auto inOrder = DimsOrder::fromValue(input);
        if (inOrder != outOrder) {
            return mlir::failure();
        }
        // get input mempermute op
        auto inputMemPermuteOp = getSupportedInputMemPermute(input);
        if (inputMemPermuteOp == nullptr) {
            continue;
        }
        // need to permute back to input of the parent mempermute
        auto inversedMemPerm = mlir::inversePermutation(inputMemPermuteOp.mem_perm());
        auto betterNumMemPermutes = checkBetterMemPerm(inversedMemPerm, bestNumNonTrivialPermutes);
        if (betterNumMemPermutes.has_value()) {
            bestMemPerm = inversedMemPerm;
            bestNumNonTrivialPermutes = betterNumMemPermutes.getValue();
        }
    }

    // try with output permute
    auto outputMemPermuteOp = getSupportedOutputMemPermute(origOp->getResult(0));
    if (outputMemPermuteOp != nullptr) {
        auto memPerm = outputMemPermuteOp.mem_perm();
        auto betterNumMemPermutes = checkBetterMemPerm(memPerm, bestNumNonTrivialPermutes);
        if (betterNumMemPermutes.has_value()) {
            bestMemPerm = memPerm;
            bestNumNonTrivialPermutes = betterNumMemPermutes.getValue();
        }
    }

    if (bestMemPerm == idMap) {
        return matchFailed(rewriter, origOp, "Already the best solution");
    }

    rewriter.startRootUpdate(origOp);
    rewriter.setInsertionPoint(origOp);

    // add permutes to inputs
    const auto origOrder = DimsOrder::fromValue(origOp->getResult(0));
    const auto newOrder = applyPermutation(origOrder, DimsOrder::fromAffineMap(bestMemPerm));
    for (auto& inputOperand : origOp->getOpOperands()) {
        auto inMemPermuteOp = rewriter.create<IE::MemPermuteOp>(origOp->getLoc(), inputOperand.get(),
                                                                newOrder.toAffineMap(ctx), bestMemPerm);
        inputOperand.set(inMemPermuteOp.output());
    }

    // change output type of layerOp
    auto output = origOp->getOpResult(0);
    const auto origType = output.getType().cast<vpux::NDTypeInterface>();
    const auto newType = inferNewTypeWithMemPerm(origType, bestMemPerm, newOrder);
    output.setType(newType);

    // add permutes to output
    rewriter.setInsertionPointAfter(origOp);
    auto outMemPermuteOp = rewriter.create<IE::MemPermuteOp>(origOp->getLoc(), output, origOrder.toAffineMap(ctx),
                                                             mlir::inversePermutation(bestMemPerm));
    output.replaceAllUsesExcept(outMemPermuteOp.output(), outMemPermuteOp);

    rewriter.finalizeRootUpdate(origOp);

    return mlir::success();
}

//
// AdjustForTile
//
// This pattern tries to move the permutes after tileOp up if it will become
// a trivial permute after the adjustment
class AdjustForTile final : public mlir::OpRewritePattern<IE::TileOp> {
public:
    AdjustForTile(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::TileOp>(ctx), _log(log) {
        setDebugName("AdjustForTile");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TileOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AdjustForTile::matchAndRewrite(IE::TileOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto ctx = getContext();

    if (!origOp->hasOneUse()) {
        return matchFailed(rewriter, origOp, "TileOp has multiple uses");
    }

    auto outputPermuteOp = mlir::dyn_cast_or_null<IE::MemPermuteOp>(*origOp->getUsers().begin());
    if (outputPermuteOp == nullptr) {
        return matchFailed(rewriter, origOp, "No MemPermuteOp found");
    }

    auto tileInType = origOp.input().getType().cast<NDTypeInterface>();
    auto tileInMemShape = tileInType.getMemShape();

    auto memPerm = outputPermuteOp.mem_perm();
    if (!isTrivialPermute(tileInMemShape, memPerm)) {
        return matchFailed(rewriter, origOp, "Not beneficial moving MemPermute up");
    }

    auto repeatsValues = origOp.repeats_values();
    if (!repeatsValues.has_value()) {
        return matchFailed(rewriter, origOp, "No repeats values found, please run canonicalizer before this pass");
    }

    auto dstOrder = DimsOrder::fromAffineMap(outputPermuteOp.dst_order());
    auto newPermuteOutType = inferNewTypeWithMemPerm(tileInType, memPerm, dstOrder);
    auto newPermuteOp = rewriter.create<IE::PermuteCastOp>(outputPermuteOp->getLoc(), newPermuteOutType, origOp.input(),
                                                           dstOrder.toAffineMap(ctx), memPerm);

    auto origOrder = tileInType.getDimsOrder();
    auto repeatsOnOrigShape = Shape(parseIntArrayAttr<int64_t>(repeatsValues.getValue()));
    auto repeatsOnOrigMemShape = origOrder.toMemoryOrder(repeatsOnOrigShape);
    auto repeatsOnNewMemShape = applyPerm(repeatsOnOrigMemShape, memPerm);
    auto repeatsOnNewShape = dstOrder.toLogicalOrder(repeatsOnNewMemShape);
    auto newTileOutType = outputPermuteOp.output().getType();
    auto newTileOp = rewriter.create<IE::TileOp>(origOp->getLoc(), newTileOutType, newPermuteOp.output(), nullptr,
                                                 getIntArrayAttr(ctx, repeatsOnNewShape));

    outputPermuteOp.replaceAllUsesWith(newTileOp.output());

    return mlir::success();
}

//
// AdjustMemPermuteAroundOpPass
//

class AdjustMemPermuteAroundOpPass final : public IE::AdjustMemPermuteAroundOpBase<AdjustMemPermuteAroundOpPass> {
public:
    explicit AdjustMemPermuteAroundOpPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AdjustMemPermuteAroundOpPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AdjustForEltwise>(&ctx, _log);
    patterns.add<AdjustForTile>(&ctx, _log);
    IE::MemPermuteOp::getCanonicalizationPatterns(patterns, &ctx);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createAdjustMemPermuteAroundOpPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustMemPermuteAroundOpPass(Logger log) {
    return std::make_unique<AdjustMemPermuteAroundOpPass>(log);
}
