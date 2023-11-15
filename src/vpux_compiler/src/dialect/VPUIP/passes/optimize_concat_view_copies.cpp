//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
// AvoidConcatExtraChannel
//
class AvoidConcatExtraChannel : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    AvoidConcatExtraChannel(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::LogicalResult checkConcatUsers(mlir::Value output, int64_t& channels, int64_t& channelOffsets,
                                         SmallVector<VPUIP::SubViewOp>& subViewOps) const;
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
mlir::LogicalResult AvoidConcatExtraChannel::checkConcatUsers(mlir::Value output, int64_t& channels,
                                                              int64_t& channelOffsets,
                                                              SmallVector<VPUIP::SubViewOp>& subViewOps) const {
    auto subviews = output.getUsers();
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

        if (subview.static_strides().has_value()) {
            return mlir::failure();
        }

        if (channelOffsets != 0 && channelOffsets != offsets[Dims4D::Act::C.ind()]) {
            return mlir::failure();
        }

        if (channels != 0 && channels != sizes[Dims4D::Act::C.ind()]) {
            return mlir::failure();
        }

        channels = sizes[Dims4D::Act::C.ind()];
        channelOffsets = offsets[Dims4D::Act::C.ind()];
        subViewOps.push_back(subview);
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
mlir::LogicalResult AvoidConcatExtraChannel::checkConcatInputs(
        mlir::ValueRange inputs, int64_t& channels, SmallVector<VPUIP::NCEClusterTilingOp>& tilingCopies) const {
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

mlir::Operation* AvoidConcatExtraChannel::createAlloc(mlir::PatternRewriter& rewriter, VPUIP::NCEClusterTilingOp copyOp,
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
mlir::LogicalResult AvoidConcatExtraChannel::matchAndRewrite(VPUIP::ConcatViewOp concatOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("Got VPUIP.ConcatViewOp at '{0}'", concatOp->getLoc());
    auto nestedLogger = _log.nest();

    auto concatOutput = concatOp.output();
    int64_t numChannels = 0;
    int64_t channelOffsets = 0;
    SmallVector<VPUIP::SubViewOp> subViewOps;
    if (checkConcatUsers(concatOutput, numChannels, channelOffsets, subViewOps).failed()) {
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
    newOffset[Dims4D::Act::C.ind()] = channelOffsets;

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

    for (auto& subviewOp : subViewOps) {
        auto newOffsets = parseIntArrayAttr<int64_t>(subviewOp.static_offsetsAttr());
        newOffsets[Dims4D::Act::C.ind()] = 0;
        auto newOffsetsAttr = getIntArrayAttr(subviewOp.getContext(), makeArrayRef(newOffsets));
        subviewOp->setAttr(subviewOp.static_offsetsAttrName(), newOffsetsAttr);
    }

    for (auto copy : inputsCopies) {
        rewriter.eraseOp(copy);
    }

    nestedLogger.trace("Successfully optimized CMX->DDR->CMX pattern");
    return mlir::success();
}

//
// FuseConcatView
//

/*
    TilingCopyOp/CopyOp  ...  TilingCopyOp/CopyOp
               \                 /
                ConcatView (DDR)
                        |
                CopyOp(DDR2DDR)      TilingCopyOp/CopyOp
                        \              /
                        ConcatView (DDR)


    TilingCopyOp/CopyOp  ...  TilingCopyOp/CopyOp     TilingCopyOp/CopyOp
                     \                 |                  /
                                ConcatView (DDR)
*/

class FuseConcatView final : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    FuseConcatView(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx), _log(log) {
    }

    bool isLegalConcatViewPattern(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log) const;
    bool hasCopyOpForAllInputs(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log) const;
    bool hasOneDDR2DDRCopyWithConcatViewConsumer(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log) const;

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp concatViewOp, mlir::PatternRewriter& rewriter) const final;
    mlir::LogicalResult fuseTwoConcatViewInputs(VPUIP::ConcatViewOp concatViewOp, mlir::PatternRewriter& rewriter,
                                                vpux::Logger log) const;

private:
    Logger _log;
};

bool FuseConcatView::hasCopyOpForAllInputs(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log) const {
    log.nest().trace("Checking hasCopyOpForAllInputs");

    auto isCopyOpWithSingleUser = [&log](mlir::Operation* op) {
        if (auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(op)) {
            if (!mlir::isa<VPUIP::SubViewOp>(copyOp.output_buff().getDefiningOp())) {
                log.nest().nest().trace("Parent CopyOp's output buffer is not defined by a SubViewOp: '{0}'",
                                        copyOp->getLoc());
                return false;
            }

            return copyOp.output().hasOneUse();
        }

        if (auto clusterCopyOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(op)) {
            if (!mlir::isa<VPUIP::CopyOp>(clusterCopyOp.getInnerTaskOp())) {
                log.nest().nest().trace("ConcatView input is not Copy op: '{0}'", clusterCopyOp->getLoc());
                return false;
            }

            if (!mlir::isa<VPUIP::SubViewOp>(clusterCopyOp.output_buffs()[0].getDefiningOp())) {
                log.nest().nest().trace(
                        "Parent NCEClusterTilingOp's output buffer is not defined by a SubViewOp: '{0}'",
                        clusterCopyOp->getLoc());
                return false;
            }

            return clusterCopyOp->hasOneUse();
        }

        return false;
    };

    return llvm::all_of(concatViewOp.inputs(), [&](auto input) {
        return isCopyOpWithSingleUser(input.getDefiningOp());
    });
}

bool FuseConcatView::hasOneDDR2DDRCopyWithConcatViewConsumer(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log) const {
    log.nest().trace("Checking hasOneDDR2DDRCopyWithConcatViewConsumer");

    if (!concatViewOp.output().hasOneUse()) {
        log.nest().nest().trace("ConcatView Op has more than one user");
        return false;
    }

    auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(*concatViewOp.output().getUsers().begin());
    if (!copyOp) {
        log.nest().nest().trace("Consumer of ConcatView Op is not Copy Op");
        return false;
    }

    if (!copyOp.output().hasOneUse()) {
        log.nest().nest().trace("CopyOp Op no user or has more than one user");
        return false;
    }

    if (!mlir::isa<VPUIP::ConcatViewOp>(*copyOp.output().getUsers().begin())) {
        log.nest().nest().trace("Consumer of Copy Op is not ConcatView Op");
        return false;
    }

    return VPUIP::isCopyFromDDR(copyOp) && VPUIP::isCopyToDDR(copyOp);
}

// Fuse ConcatView Ops to remove unnecessary copies, two conditions need to be satisfied:
// a) The Stride Level for each ConcatView input (after fusing) should be no more than 2;
//     It's a runtime and HW limitation in order to get the right NNDMA descriptor, we support a maximum of 3D DMA
//     transfers with 2 levels of striding.
// b) The number of inputs from the second ConcatView, which come from the output of the first should no more than 1;
//     For example, first ConcatView has M inputs, second ConcatView has N inputs, out of which P of them are the output
//     of the first ConcatView After fusing, the number of input copies is: M * P + (N - P)
//     Can't ensure we get benefit when P is of a large size. Limit optimization to P=1.
bool FuseConcatView::isLegalConcatViewPattern(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log) const {
    if (concatViewOp.output().use_empty()) {
        log.nest().trace("Cannot find user copy op at '{0}'", concatViewOp->getLoc());
        return false;
    }

    if (!hasCopyOpForAllInputs(concatViewOp, log)) {
        log.nest().trace("Not all inputs is CopyOp for first ConcatViewOp at '{0}'", concatViewOp->getLoc());
        return false;
    }

    if (!hasOneDDR2DDRCopyWithConcatViewConsumer(concatViewOp, log)) {
        log.nest().trace("Not only one user is DDR2DDR copy with ConcatViewOp for op at '{0}'", concatViewOp->getLoc());
        return false;
    }

    log.nest().trace("FuseConcatView: Found legal ConcatView pattern at op '{0}'", concatViewOp->getLoc());

    return true;
}

mlir::LogicalResult FuseConcatView::fuseTwoConcatViewInputs(VPUIP::ConcatViewOp concatViewOp,
                                                            mlir::PatternRewriter& rewriter, vpux::Logger log) const {
    // Get current concat's memref.alloc op, which will be removed
    auto firstConcatMemAlloc = VPUIP::getRootAlloc<mlir::memref::AllocOp>(concatViewOp.output_buff());
    if (firstConcatMemAlloc == nullptr) {
        log.nest().trace("Cannot rewrite because current concat '{0}' output isn't master buffer",
                         concatViewOp->getLoc());
        return mlir::failure();
    }

    // Get Copy and next ConcatView Op
    auto outputCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(*concatViewOp.output().getUsers().begin());
    VPUX_THROW_UNLESS(outputCopyOp != nullptr, "Cannot get DDR to DDR Copy Op after '{0}'", concatViewOp->getLoc());
    VPUIP::SubViewOp outCopySubView = outputCopyOp.output_buff().getDefiningOp<VPUIP::SubViewOp>();

    auto nextConcatViewOp = mlir::dyn_cast<VPUIP::ConcatViewOp>(*outputCopyOp.output().getUsers().begin());
    VPUX_THROW_UNLESS(nextConcatViewOp != nullptr, "Cannot get second ConcatView Op");

    auto nextConcatMemAlloc = VPUIP::getRootAlloc<mlir::memref::AllocOp>(nextConcatViewOp.output_buff());
    if (nextConcatMemAlloc == nullptr) {
        log.nest().trace("Cannot rewrite because next concat '{0}' output isn't master buffer",
                         nextConcatViewOp->getLoc());
        return mlir::failure();
    }

    // Create an array of the new input copy ops
    SmallVector<mlir::Value> newCopyInputs;
    SmallVector<mlir::Value> oldCopyInputs;
    SmallVector<VPUIP::SubViewOp> oldSubViewInputs;
    newCopyInputs.reserve(concatViewOp.inputs().size() + nextConcatViewOp.inputs().size() - 1);
    oldCopyInputs.reserve(concatViewOp.inputs().size());
    oldSubViewInputs.reserve(concatViewOp.inputs().size());

    auto isStrideConcat = [](VPUIP::SubViewOp subView) {
        if (subView.static_stridesAttr() == nullptr) {
            return false;
        }

        auto strides = parseIntArrayAttr<int64_t>(subView.static_stridesAttr());
        return llvm::any_of(strides, [](auto stride) {
            return stride > 1;
        });
    };

    for (size_t nextInIdx = 0; nextInIdx < nextConcatViewOp.inputs().size(); ++nextInIdx) {
        auto siblingCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(nextConcatViewOp.inputs()[nextInIdx].getDefiningOp());
        if (!(siblingCopyOp && siblingCopyOp == outputCopyOp)) {
            newCopyInputs.push_back(nextConcatViewOp.inputs()[nextInIdx]);
            continue;
        }

        SmallVector<int64_t> outCopyOffsets = parseIntArrayAttr<int64_t>(outCopySubView.static_offsetsAttr());
        SmallVector<int64_t> outCopySizes = parseIntArrayAttr<int64_t>(outCopySubView.static_sizesAttr());
        if (isStrideConcat(outCopySubView)) {
            log.nest().trace("Fusing Concat Op with stride has no performance benefits");
            return mlir::failure();
        }

        for (size_t firstInIdx = 0; firstInIdx < concatViewOp.inputs().size(); ++firstInIdx) {
            auto op = concatViewOp.inputs()[firstInIdx].getDefiningOp();

            bool isClusterCopy = false;
            VPUIP::SubViewOp inCopySubView;
            auto inCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(op);
            if (inCopyOp) {
                inCopySubView = inCopyOp.output_buff().getDefiningOp<VPUIP::SubViewOp>();
            }
            auto clusterCopyOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(op);
            if (clusterCopyOp) {
                inCopySubView = clusterCopyOp.output_buffs()[0].getDefiningOp<VPUIP::SubViewOp>();
                isClusterCopy = true;
            }

            VPUX_THROW_WHEN(inCopySubView == nullptr, "Cannot get SubViewOp");
            oldCopyInputs.push_back(concatViewOp.inputs()[firstInIdx]);
            oldSubViewInputs.push_back(inCopySubView);

            SmallVector<int64_t> inCopyOffsets = parseIntArrayAttr<int64_t>(inCopySubView.static_offsetsAttr());
            SmallVector<int64_t> inCopySizes = parseIntArrayAttr<int64_t>(inCopySubView.static_sizesAttr());

            if (isStrideConcat(inCopySubView)) {
                log.nest().trace("Fusing Concat Op with stride has no performance benefits");
                return mlir::failure();
            }

            VPUX_THROW_WHEN(outCopyOffsets.size() != inCopyOffsets.size() || outCopySizes.size() != inCopySizes.size(),
                            "Input and output copy subviews have different-sized attributes");

            SmallVector<int64_t> newCopyOffsets(outCopyOffsets.size());
            SmallVector<int64_t> newCopySizes(outCopySizes.size());

            for (size_t idx = 0; idx < newCopyOffsets.size(); ++idx) {
                newCopySizes[idx] = inCopySizes[idx];
                newCopyOffsets[idx] = outCopyOffsets[idx] + inCopyOffsets[idx];
            }

            auto newSubViewOp = rewriter.create<VPUIP::SubViewOp>(outCopySubView->getLoc(), outCopySubView.source(),
                                                                  newCopyOffsets, newCopySizes);
            if (newSubViewOp->isBeforeInBlock(nextConcatMemAlloc)) {
                if (auto groupOp = mlir::dyn_cast<vpux::GroupedViewOpInterface>(nextConcatMemAlloc)) {
                    for (auto source : groupOp.getViewSources()) {
                        source.getDefiningOp()->moveBefore(newSubViewOp);
                    }
                }
                nextConcatMemAlloc->moveBefore(newSubViewOp);
            }

            if (isClusterCopy) {
                const auto copyOutBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                                    mlir::ValueRange newOperands) {
                    builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
                };

                SmallVector<mlir::Value> inputOutputOperands = {op->getOperand(0), newSubViewOp.result()};
                auto newCopyInCluster = rewriter.create<VPUIP::NCEClusterTilingOp>(
                        op->getLoc(), newSubViewOp->getResult(0).getType(), inputOutputOperands, copyOutBodyBuilder);

                auto innerCopyOp = newCopyInCluster.getInnerTaskOpOfType<VPUIP::CopyOp>();
                if (!VPUIP::hasLegalStridingLevel(innerCopyOp)) {
                    log.nest().trace("DMA Striding Level is illegal. Fusing Concat Op have no benefit");
                    rewriter.eraseOp(newCopyInCluster);
                    rewriter.eraseOp(newSubViewOp);
                    return mlir::failure();
                }
                newCopyInputs.push_back(newCopyInCluster->getResult(0));
                continue;
            }

            auto newCopyOp = rewriter.create<VPUIP::CopyOp>(op->getLoc(), op->getOperand(0), newSubViewOp.result());
            if (!VPUIP::hasLegalStridingLevel(newCopyOp)) {
                log.nest().trace("DMA Striding Level is illegal. Fusing Concat Op have no benefit");
                rewriter.eraseOp(newCopyOp);
                rewriter.eraseOp(newSubViewOp);
                return mlir::failure();
            }
            newCopyInputs.push_back(newCopyOp.output());
        }
    }

    rewriter.setInsertionPoint(nextConcatViewOp);
    rewriter.replaceOpWithNewOp<VPUIP::ConcatViewOp>(nextConcatViewOp, nextConcatViewOp.output().getType(),
                                                     newCopyInputs, nextConcatViewOp.output_buff());

    // Erase the old hanging structure
    rewriter.eraseOp(outputCopyOp);
    rewriter.eraseOp(outCopySubView);
    rewriter.eraseOp(concatViewOp);

    for (size_t inIdx = 0; inIdx < oldCopyInputs.size(); ++inIdx) {
        rewriter.eraseOp(oldCopyInputs[inIdx].getDefiningOp());
        rewriter.eraseOp(oldSubViewInputs[inIdx]);
    }

    rewriter.eraseOp(firstConcatMemAlloc);

    return mlir::success();
}

mlir::LogicalResult FuseConcatView::matchAndRewrite(VPUIP::ConcatViewOp concatViewOp,
                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("FuseConcatView: Got ConcatView Op at '{0}'", concatViewOp.getLoc());

    if (!isLegalConcatViewPattern(concatViewOp, _log)) {
        _log.nest().trace("FuseConcatView: Cannot rewrite this concat Op");
        return mlir::failure();
    }

    return fuseTwoConcatViewInputs(concatViewOp, rewriter, _log);
}

// RemoveDDRToDDRCopyAfterConcatView
//

/*
            CopyOp     ...      CopyOp
               \                 /
                ConcatView (DDR)
                        |
                (Pure View Ops)
                        |
                CopyOp(DDR2DDR)

Optimized:
            CopyOp     ...      CopyOp
               \                 /
                ConcatView (DDR)
                        |
                (Pure View Ops)
*/

class RemoveDDRToDDRCopyAfterConcatView final : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    RemoveDDRToDDRCopyAfterConcatView(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx), _log(log) {
    }

    mlir::Operation* getTargetCopyOp(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log) const;

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp concatViewOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::Operation* RemoveDDRToDDRCopyAfterConcatView::getTargetCopyOp(VPUIP::ConcatViewOp concatViewOp,
                                                                    vpux::Logger log) const {
    log.nest().trace("Checking ConcatView Copy pattern");

    auto childOp = *concatViewOp.output().getUsers().begin();
    while (childOp != nullptr && mlir::isa<VPUIP::GenericReshapeOp, VPUIP::PermuteCastOp, VPUIP::QuantizeCastOp,
                                           VPUIP::ShapeCastOp, VPUIP::CopyOp>(childOp)) {
        if (!childOp->getResult(0).hasOneUse()) {
            log.nest().trace("child op user does not match");
            return nullptr;
        } else if (mlir::isa<VPUIP::CopyOp>(childOp)) {
            return childOp;
        } else {
            childOp = *childOp->getResult(0).getUsers().begin();
        }
    }
    log.nest().trace("Could not find ConcatView Copy pattern");
    return nullptr;
}

mlir::LogicalResult RemoveDDRToDDRCopyAfterConcatView::matchAndRewrite(VPUIP::ConcatViewOp concatViewOp,
                                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("RemoveDDRToDDRCopyAfterConcatView: Got ConcatView Op at '{0}'", concatViewOp.getLoc());

    if (!concatViewOp.output().hasOneUse()) {
        _log.nest().trace("RemoveDDRToDDRCopyAfterConcatView: Only support ConcatView has one user");
        return mlir::failure();
    }
    auto targetOp = getTargetCopyOp(concatViewOp, _log);
    if (targetOp == nullptr) {
        _log.nest().trace("RemoveDDRToDDRCopyAfterConcatView: Cannot find the target Copy Op");
        return mlir::failure();
    }
    auto targetCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(targetOp);
    if (!VPUIP::isCopyToDDR(targetCopyOp) || !VPUIP::isCopyFromDDR(targetCopyOp)) {
        _log.nest().trace("RemoveDDRToDDRCopyAfterConcatView: Target Copy Op is not from DDR to DDR");
        return mlir::failure();
    }

    // Check if the CopyOp copies to output
    if (targetCopyOp.output_buff().isa<mlir::BlockArgument>()) {
        _log.trace("RemoveDDRToDDRCopyAfterConcatView: Cannot rewrite because it is last copy");
        return mlir::failure();
    }

    VPUIP::SubViewOp outCopySubView = targetCopyOp.output_buff().getDefiningOp<VPUIP::SubViewOp>();
    if (outCopySubView != nullptr) {
        _log.nest().trace("Cannot remove copy op with subView");
        return mlir::failure();
    }

    targetCopyOp.output().replaceAllUsesWith(targetCopyOp.input());
    rewriter.eraseOp(targetCopyOp);
    _log.trace("Successfully removed redundant copy Op after ConcatView");
    return mlir::success();
}

//
// OptimizeConcatViewCopiesPass
//

class OptimizeConcatViewCopiesPass final : public VPUIP::OptimizeConcatViewCopiesBase<OptimizeConcatViewCopiesPass> {
public:
    explicit OptimizeConcatViewCopiesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void OptimizeConcatViewCopiesPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AvoidConcatExtraChannel>(&ctx, _log);
    patterns.add<FuseConcatView>(&ctx, _log);
    patterns.add<RemoveDDRToDDRCopyAfterConcatView>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeConcatViewCopiesPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createOptimizeConcatViewCopiesPass(Logger log) {
    return std::make_unique<OptimizeConcatViewCopiesPass>(log);
}
