//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// OptimizeConcat
//
class OptimizeConcat : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    OptimizeConcat(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::LogicalResult checkConcatUsers(mlir::Value::user_range users, int64_t& channels) const;
    mlir::LogicalResult checkConcatInputs(mlir::ValueRange inputs, int64_t& channels,
                                          SmallVector<VPUIP::NCEClusterTilingOp>& tilingCopies) const;

    mlir::Operation* createAlloc(mlir::PatternRewriter& rewriter, VPUIP::NCEClusterTilingOp copyOp,
                                 int64_t channels) const;

    Logger _log;
};

/*
 Check if all Concat users are Subview with same channels slice
 less than Concat channels
                          Concat (m output channels)
        |                                        |
    Subview (n output channels)              Subview (n output channels)
    m > n

*/
mlir::LogicalResult OptimizeConcat::checkConcatUsers(mlir::Value::user_range subviews, int64_t& channels) const {
    if (subviews.empty()) {
        return mlir::failure();
    }

    for (const auto user : subviews) {
        auto subview = mlir::dyn_cast_or_null<VPUIP::SubViewOp>(user);

        if (subview == nullptr) {
            return mlir::failure();
        }

        auto offsets = parseIntArrayAttr<int64_t>(subview.static_offsetsAttr());
        auto sizes = parseIntArrayAttr<int64_t>(subview.static_sizesAttr());

        if (offsets[Dims4D::Act::C.ind()] != 0) {
            return mlir::failure();
        }

        if (channels != 0 && channels != sizes[Dims4D::Act::C.ind()]) {
            return mlir::failure();
        }

        channels = sizes[Dims4D::Act::C.ind()];
    }

    return mlir::success();
}

/*
 Check if all Concat inputs copy NCE result with more channels
 than Subview after Concat
 Concat joins its inputs not by channel dimension

            Op                              Op
            |                                 |
        TilingCopy (m output channels)     TilingCopy (m output channels)
            |                                 |
          Subview (m output channels)       Subview (m output channels)
            |                                 |
                                Concat (m output channels)
                                   |
                                Subview (n output channels)
    m > n

*/
mlir::LogicalResult OptimizeConcat::checkConcatInputs(mlir::ValueRange inputs, int64_t& channels,
                                                      SmallVector<VPUIP::NCEClusterTilingOp>& tilingCopies) const {
    if (inputs.empty()) {
        return mlir::failure();
    }

    for (auto input : inputs) {
        auto tilingCopy = input.getDefiningOp<VPUIP::NCEClusterTilingOp>();

        if (tilingCopy == nullptr || tilingCopy.getInnerTaskOpOfType<VPUIP::CopyOp>() == nullptr) {
            return mlir::failure();
        }

        if (!tilingCopy->getResult(0).hasOneUse()) {
            return mlir::failure();
        }

        auto copyOpOutput = tilingCopy.getOutputs()[0];

        auto subview = copyOpOutput.getDefiningOp<VPUIP::SubViewOp>();

        if (subview == nullptr) {
            return mlir::failure();
        }

        auto offsets = parseIntArrayAttr<int64_t>(subview.static_offsetsAttr());
        auto sizes = parseIntArrayAttr<int64_t>(subview.static_sizesAttr());

        if (offsets[Dims4D::Act::C.ind()] != 0) {
            return mlir::failure();
        }

        if (channels >= sizes[Dims4D::Act::C.ind()]) {
            return mlir::failure();
        }

        if (VPUIP::getRootAlloc<mlir::memref::AllocOp>(subview.source()) == nullptr) {
            return mlir::failure();
        }

        tilingCopies.push_back(tilingCopy);
    }

    return mlir::success();
}

mlir::Operation* OptimizeConcat::createAlloc(mlir::PatternRewriter& rewriter, VPUIP::NCEClusterTilingOp copyOp,
                                             int64_t channels) const {
    auto copyOpOutput = copyOp.getOutputs()[0];

    auto subview = copyOpOutput.getDefiningOp<VPUIP::SubViewOp>();

    auto opOutputType = subview.source().getType().cast<vpux::NDTypeInterface>();
    auto sourceShape = opOutputType.getShape().toValues();
    sourceShape[Dims4D::Act::C] = channels;
    auto newOpOutputType = opOutputType.changeShape(ShapeRef(sourceShape));

    return allocateBuffersOfType(_log, copyOp->getLoc(), rewriter, newOpOutputType).front().getDefiningOp();
}

/*

            Op                                Op
            |                                 |
        TilingCopy (m output channels)     TilingCopy (m output channels)
            |                                 |
          Subview (m output channels)       Subview (m output channels)
            |                                 |
                                Concat (m output channels)
                                   |
                                Subview (n output channels)
    m > n

    is converted to pattern

            Op (m output channels)           Op (m output channels)
            |                                 |
          Subview (n output channels)       Subview (n output channels)
            |                                 |
        TilingCopy (n output channels)     TilingCopy (n output channels)
            |                                 |
                                Concat (n output channels)

*/
mlir::LogicalResult OptimizeConcat::matchAndRewrite(VPUIP::ConcatViewOp concatOp,
                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("Got VPUIP.ConcatViewOp at '{0}'", concatOp->getLoc());
    auto nestedLogger = _log.nest();

    auto concatUsers = concatOp.output().getUsers();
    int64_t numChannels = 0;
    if (checkConcatUsers(concatUsers, numChannels).failed()) {
        nestedLogger.trace("Cannot optimize because of users requirements");
        return mlir::failure();
    }

    if (numChannels == 0 || numChannels >= getShape(concatOp.output())[Dims4D::Act::C]) {
        nestedLogger.trace("Cannot optimize because of channels requirements");
        return mlir::failure();
    }

    auto inputs = concatOp.inputs();
    SmallVector<VPUIP::NCEClusterTilingOp> inputsCopies;
    inputsCopies.reserve(inputs.size());
    if (checkConcatInputs(inputs, numChannels, inputsCopies).failed() || inputsCopies.empty()) {
        nestedLogger.trace("Cannot optimize because of input requirements");
        return mlir::failure();
    }

    auto* alloc = createAlloc(rewriter, inputsCopies.front(), numChannels);

    if (alloc == nullptr) {
        nestedLogger.trace("Cannot allocate new buffer");
        return mlir::failure();
    }

    SmallVector<mlir::Value> concatInputs;
    concatInputs.reserve(inputs.size());

    SmallVector<int64_t> newOffset(concatOp.getType().cast<vpux::NDTypeInterface>().getRank(), 0);

    for (auto copy : inputsCopies) {
        auto copyOpInput = copy.getInputs()[0];
        auto copyOpOutput = copy.getOutputs()[0];

        auto subview = copyOpOutput.getDefiningOp<VPUIP::SubViewOp>();

        auto sizes = parseIntArrayAttr<int64_t>(subview.static_sizesAttr());
        sizes[Dims4D::Act::C.ind()] = numChannels;

        SmallVector<int64_t> newSizes = to_small_vector(getShape(copyOpInput).raw());
        newSizes[Dims4D::Act::C.ind()] = numChannels;

        auto newSubviewNCE = rewriter.create<VPUIP::SubViewOp>(
                subview.getLoc(), copyOpInput, getIntArrayAttr(subview.getContext(), makeArrayRef(newOffset)),
                getIntArrayAttr(subview.getContext(), makeArrayRef(newSizes)));

        auto newSubviewCopy =
                rewriter.create<VPUIP::SubViewOp>(subview.getLoc(), alloc->getResult(0), subview.static_offsetsAttr(),
                                                  getIntArrayAttr(subview.getContext(), makeArrayRef(sizes)));

        const auto copyOutBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                            mlir::ValueRange newOperands) {
            builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
        };

        SmallVector<mlir::Value> inputsOutputOperands = {newSubviewNCE, newSubviewCopy};
        auto newTilingCopy = rewriter.create<VPUIP::NCEClusterTilingOp>(copy->getLoc(), newSubviewCopy.getType(),
                                                                        inputsOutputOperands, copyOutBodyBuilder);

        concatInputs.push_back(newTilingCopy.results()[0]);
    }

    auto newOp = rewriter.replaceOpWithNewOp<VPUIP::ConcatViewOp>(concatOp, concatInputs, alloc->getResult(0));

    for (auto user : newOp.output().getUsers()) {
        vpux::inferReturnTypes(user, vpux::InferShapedTypeMode::ALL);
    }

    for (auto copy : inputsCopies) {
        rewriter.eraseOp(copy);
    }
    nestedLogger.trace("Successfully optimized CMX->DDR->CMX pattern");
    return mlir::success();
}

//
// OptimizeSpillingCopiesPass
//

class OptimizeSpillingCopiesPass final : public VPUIP::OptimizeSpillingCopiesBase<OptimizeSpillingCopiesPass> {
public:
    explicit OptimizeSpillingCopiesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void OptimizeSpillingCopiesPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<OptimizeConcat>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeSpillingCopiesPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createOptimizeSpillingCopiesPass(Logger log) {
    return std::make_unique<OptimizeSpillingCopiesPass>(log);
}
