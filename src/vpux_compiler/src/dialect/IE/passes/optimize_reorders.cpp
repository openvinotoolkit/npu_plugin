//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/adjust_layout_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/IE/prefix.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// ReorderWithShapeChange
//
template <class ConcreteOp>
class ReorderWithShapeChange final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    ReorderWithShapeChange(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
        this->setDebugName("ReorderWithShapeChange");
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origReshapeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// This function is to find groups of axes that are reshaped
// For example, (3,16,8,2)#NCHW -> Reshape -> (1,48,4,4)#NCHW
// The result will be {{N, C}, {H, W}}
SmallVector<SmallVector<Dim>> getReshapedAxes(ShapeRef inShape, ShapeRef outShape, DimsOrder order) {
    SmallVector<SmallVector<Dim>> reshapedAxes;
    SmallVector<Dim> reshapedGroup;
    int64_t inProduct;
    int64_t outProduct;
    bool startMatch = false;
    for (const auto& dim : order.toPermutation()) {
        auto inDimSize = inShape[dim];
        auto outDimSize = outShape[dim];
        if (startMatch) {
            // Keep iterating dims until the total element sizes are the same, and those dims form a group
            reshapedGroup.push_back(dim);
            inProduct *= inDimSize;
            outProduct *= outDimSize;
            if (inProduct == outProduct) {
                reshapedAxes.push_back(reshapedGroup);
                startMatch = false;
            }
        } else {
            // Start matching if dim sizes are different
            if (inDimSize != outDimSize) {
                reshapedGroup.assign({dim});
                inProduct = inDimSize;
                outProduct = outDimSize;
                startMatch = true;
            }
        }
    }
    VPUX_THROW_UNLESS(startMatch == false, "Reshape's input {0} and output {1} are not matched", inShape, outShape);

    return reshapedAxes;
}

template <class ConcreteOp>
mlir::LogicalResult ReorderWithShapeChange<ConcreteOp>::matchAndRewrite(ConcreteOp origReshapeOp,
                                                                        mlir::PatternRewriter& rewriter) const {
    const auto origReshapeInput = origReshapeOp->getOperand(0);
    auto origReorderOp = origReshapeInput.template getDefiningOp<IE::ReorderOp>();
    if (origReorderOp == nullptr) {
        return mlir::failure();
    }
    _log.trace("Got Reorder at '{0}' -> {1} at '{2}' pair", origReorderOp->getLoc(), origReshapeOp->getName(),
               origReshapeOp->getLoc());

    const auto origReshapeOutput = origReshapeOp->getResult(0);
    const auto inOrder = DimsOrder::fromValue(origReorderOp.input());
    const auto outOrder = DimsOrder::fromAffineMap(origReorderOp.dstOrder());
    const auto reshapeOutOrder = origReshapeOutput.getType().template dyn_cast<vpux::NDTypeInterface>().getDimsOrder();
    if (outOrder != reshapeOutOrder) {
        return matchFailed(_log.nest(), rewriter, origReshapeOp,
                           "Reshape's input order {0} and output order {1} are not the same", outOrder,
                           reshapeOutOrder);
    }

    // Reorder can only propagate if the order of Reshape's reshaped axes are kept the same
    // For exmaple, (1,16,8,2)#NHWC -> Reorder -> (1,16,8,2)#NCHW -> Reshape -> (1,16,4,4)#NCHW
    // Only H & W are reshaped and their relative order stays the same
    // Thus, the Reorder can be propagated through the Reshape like below:
    // (1,16,8,2)#NHWC -> Reshape -> (1,16,4,4)#NHWC -> Reorder -> (1,16,4,4)#NCHW

    auto reshapeInShape = getShape(origReshapeInput);
    auto reshapeOutShape = getShape(origReshapeOutput);
    auto reshapedAxes = getReshapedAxes(reshapeInShape, reshapeOutShape, outOrder);

    for (const auto& group : reshapedAxes) {
        auto dimIter = group.begin();
        auto prevPos = inOrder.dimPos(*dimIter);
        while (++dimIter != group.end()) {
            auto curPos = inOrder.dimPos(*dimIter);
            if (curPos - prevPos != 1) {
                return matchFailed(
                        _log.nest(), rewriter, origReshapeOp,
                        "The orders of reshaped axes {0} - {1} are different in input order {2} and output order {3}",
                        *(dimIter - 1), *dimIter, inOrder, outOrder);
            }
            prevPos = curPos;
        }
    }

    auto shapeAttr = getIntArrayAttr(origReshapeOp.getContext(), reshapeOutShape);
    auto shapeCastOp = rewriter.create<IE::ShapeCastOp>(origReshapeOp->getLoc(), origReorderOp.input(), shapeAttr);
    rewriter.replaceOpWithNewOp<IE::ReorderOp>(origReshapeOp, shapeCastOp.result(), origReorderOp.dstOrderAttr());

    return mlir::success();
}

//
// ReorderWithSubView
//

class ReorderWithSubView final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    ReorderWithSubView(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SliceOp>(ctx), _log(log) {
        setDebugName("ReorderWithSubView");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SliceOp origSubViewOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithSubView::matchAndRewrite(IE::SliceOp origSubViewOp,
                                                        mlir::PatternRewriter& rewriter) const {
    auto origReorderOp = origSubViewOp.source().getDefiningOp<IE::ReorderOp>();
    if (origReorderOp == nullptr) {
        return mlir::failure();
    }

    // Maintain the Reorder -> PermuteCast -> Reorder chain as it can later be reduced to a single operation
    if (auto prevPermuteCastOp = origReorderOp.input().getDefiningOp<IE::PermuteCastOp>()) {
        if (auto prevReorderOp = prevPermuteCastOp.input().getDefiningOp<IE::ReorderOp>())
            return mlir::failure();
    }

    _log.trace("Got reorder at '{0}' -> subview at '{1}' pair", origReorderOp->getLoc(), origSubViewOp->getLoc());

    const auto isHomogeneousUser = [](mlir::Operation* reorderUser) -> bool {
        return mlir::isa<IE::SliceOp, IE::ConcatOp, IE::PermuteCastOp>(reorderUser);
    };

    if (!llvm::all_of(origReorderOp->getUsers(), isHomogeneousUser)) {
        return matchFailed(_log.nest(), rewriter, origSubViewOp,
                           "Reorder has more than one user and they are heterogeneous");
    }

    // In case "ReorderOp + SliceOp", if ReorderOp has no permutation for last dim (For example
    // affine_map<(d0, d2, d3, d1, d4) -> (d0, d1, d2, d3, d4)>), and the SliceOp output shape size for last dim
    // is 1 (For example, d0xd1xd2xd3x1), then the new Reorder <(d0, d2, d3, d1, 1) -> (d0, d1, d2, d3, 1)>
    // after swap will make it worse from performance perspective as inefficient DMA
    if (!origReorderOp.getResult().hasOneUse()) {
        bool allSlicesUsers = true;
        bool benifitToSwap = true;
        for (auto* reorderUser : llvm::make_early_inc_range(origReorderOp->getUsers())) {
            auto reorderUserSliceOp = mlir::dyn_cast<IE::SliceOp>(reorderUser);
            if (reorderUserSliceOp == nullptr) {
                allSlicesUsers = false;
                break;
            }

            auto reorderInputDimsOrder = DimsOrder::fromValue(origReorderOp.input());
            auto reorderOutputDimsOrder = DimsOrder::fromValue(origReorderOp.output());
            auto reorderInputPerm = reorderInputDimsOrder.toPermutation();
            auto reorderOutputPerm = reorderOutputDimsOrder.toPermutation();
            auto inputPermEnd = (reorderInputPerm.end() - 1)->ind();
            auto outputPermEnd = (reorderOutputPerm.end() - 1)->ind();
            const auto sliceShape = parseIntArrayAttr<int64_t>(reorderUserSliceOp.static_sizes());

            if (inputPermEnd == outputPermEnd && sliceShape[outputPermEnd] == 1) {
                benifitToSwap = false;
                break;
            }
        }

        if (allSlicesUsers && !benifitToSwap) {
            return mlir::failure();
        }
    }

    auto newSubViewOp =
            rewriter.create<IE::SliceOp>(origSubViewOp->getLoc(), origReorderOp.input(),
                                         origSubViewOp.static_offsetsAttr(), origSubViewOp.static_sizesAttr());

    rewriter.replaceOpWithNewOp<IE::ReorderOp>(origSubViewOp, newSubViewOp.result(), origReorderOp.dstOrderAttr());
    return mlir::success();
}

//
// ReorderWithExpand
//

class ReorderWithExpand final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    ReorderWithExpand(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("ReorderWithExpand");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp origExpandOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

void swapExpandWithReorder(mlir::PatternRewriter& rewriter, IE::ExpandOp expandOp, IE::ReorderOp origReorderOp) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(expandOp);

    auto newExpandOp = rewriter.create<IE::ExpandOp>(expandOp->getLoc(), origReorderOp.input(),
                                                     expandOp.pads_beginAttr(), expandOp.pads_endAttr());

    rewriter.replaceOpWithNewOp<IE::ReorderOp>(expandOp, newExpandOp.output(), origReorderOp.dstOrderAttr());
}

mlir::LogicalResult ReorderWithExpand::matchAndRewrite(IE::ExpandOp origExpandOp,
                                                       mlir::PatternRewriter& rewriter) const {
    auto* ctx = origExpandOp->getContext();
    auto origReorderOp = origExpandOp.input().getDefiningOp<IE::ReorderOp>();
    if (origReorderOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got reorder at '{0}' -> Expand at '{1}' pair", origReorderOp->getLoc(), origExpandOp->getLoc());

    const auto isExpand = [](mlir::Operation* reorderUser) -> bool {
        return mlir::isa<IE::ExpandOp>(reorderUser);
    };

    if (!llvm::all_of(origReorderOp->getUsers(), isExpand)) {
        return matchFailed(_log.nest(), rewriter, origExpandOp,
                           "Reorder has more than one user and they are heterogeneous");
    }

    // If Reorder is Trivial Permute, will not swap
    // Input (1x1x512x512, NCHW) -> Reorder (1x1x512x512, NHWC) -> Expand (1x16x512x512, NHWC)
    // After Swap the Reorder is not Trivial Permute
    // Input (1x1x512x512, NCHW) -> Expand (1x16x512x512, NCHW) -> Reorder (1x16x512x512, NHWC)
    if (isTrivialReorder(origReorderOp)) {
        return mlir::failure();
    }

    // If after swap Reorder cannot support by PermuteDMA, will not swap
    // Input (1x1x512x512, NCWH) -> Reorder (1x1x512x512, NHWC) -> Expand (1x16x512x512, NHWC)
    // After Swap the Reorder cannot convert to PermuteDMA
    // Input (1x1x512x512, NCWH) -> Expand (1x16x512x512, NCWH) -> Reorder (1x16x512x512, NHWC)
    auto expandOutType = origExpandOp.output().getType().cast<vpux::NDTypeInterface>();
    auto newOrderInType = expandOutType.changeDimsOrder(DimsOrder::fromValue(origReorderOp.input()));
    auto newOrderOutType = expandOutType.changeDimsOrder(DimsOrder::fromValue(origReorderOp.output()));
    auto memPerm = getPermutationFromOrders(newOrderInType.getDimsOrder(), newOrderOutType.getDimsOrder(), ctx);
    auto unsupportPermuteDMA = [&]() -> bool {
        const auto inShape = newOrderInType.getShape();
        return newOrderInType.getRank() == 4 && memPerm == mlir::AffineMap::getPermutationMap({0, 3, 2, 1}, ctx) &&
               inShape[Dims4D::Act::C] > 1 && inShape[Dims4D::Act::H] > 1 && inShape[Dims4D::Act::W] > 1;
    };

    if (unsupportPermuteDMA()) {
        return mlir::failure();
    }

    for (auto* reorderUser : llvm::make_early_inc_range(origReorderOp->getUsers())) {
        auto expandOp = mlir::cast<IE::ExpandOp>(reorderUser);
        swapExpandWithReorder(rewriter, expandOp, origReorderOp);
    }

    return mlir::success();
}

//
// ReorderWithSplit
//

class ReorderWithSplit final : public mlir::OpRewritePattern<IE::SplitOp> {
public:
    ReorderWithSplit(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SplitOp>(ctx), _log(log) {
        setDebugName("ReorderWithSplit");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SplitOp origSplitOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithSplit::matchAndRewrite(IE::SplitOp origSplitOp, mlir::PatternRewriter& rewriter) const {
    if (origSplitOp.axis() != nullptr) {
        return mlir::failure();
    }

    auto origReorderOp = origSplitOp.input().getDefiningOp<IE::ReorderOp>();
    if (origReorderOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got reorder at '{0}' -> Split at '{1}' pair", origReorderOp->getLoc(), origSplitOp->getLoc());

    const auto inOrder = DimsOrder::fromValue(origReorderOp.input());
    const auto outOrder = DimsOrder::fromValue(origReorderOp.output());
    const auto dstOrderAttr = origReorderOp.dstOrderAttr();
    auto newSplit = rewriter.create<IE::SplitOp>(origSplitOp->getLoc(), origReorderOp.input(), origSplitOp.axis(),
                                                 origSplitOp.num_splitsAttr(), origSplitOp.axis_valueAttr());

    SmallVector<mlir::Value> newOutputs;
    newOutputs.reserve(origSplitOp.outputs().size());

    for (auto res : origSplitOp.outputs()) {
        if (res.getUses().empty()) {
            newOutputs.push_back(newSplit.getResult(res.getResultNumber()));
            continue;
        }

        _log.trace("Insert reorder '{0}' -> '{1}' for Split output at idx='{2}'.", inOrder, outOrder,
                   res.getResultNumber());
        auto reorder = rewriter.create<IE::ReorderOp>(origSplitOp->getLoc(), newSplit.getResult(res.getResultNumber()),
                                                      dstOrderAttr);
        newOutputs.push_back(reorder);
    }

    _log.trace("Replace Split with new output values.");
    rewriter.replaceOp(origSplitOp, newOutputs);

    return mlir::success();
}

//
// ReorderWithConcat
//

mlir::Value insertReorderForBlockArg(mlir::Value arg, DimsOrder dstOrder, mlir::PatternRewriter& rewriter,
                                     mlir::Location loc) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(arg.getParentBlock());
    auto reorderOp = rewriter.create<IE::ReorderOp>(loc, arg, dstOrder.toAffineMap(rewriter.getContext()));
    arg.replaceAllUsesExcept(reorderOp.output(), llvm::SmallPtrSet<mlir::Operation*, 1>{reorderOp});
    return reorderOp.output();
}

void replaceChildReorderWithNewConcat(IE::ConcatOp& origConcatOp, IE::ConcatOp& newConcatOp,
                                      mlir::PatternRewriter& rewriter, Logger log) {
    auto outputConcat = origConcatOp.output();
    auto childReorderOp = mlir::dyn_cast_or_null<IE::ReorderOp>(*outputConcat.getUsers().begin());

    auto newOutputConcat = newConcatOp.output();
    vpux::changeDimsOrder(newOutputConcat, DimsOrder::fromValue(childReorderOp.output()), log);
    childReorderOp.output().replaceAllUsesExcept(newOutputConcat, llvm::SmallPtrSet<mlir::Operation*, 1>{newConcatOp});

    rewriter.eraseOp(childReorderOp);
    rewriter.eraseOp(origConcatOp);
}

//
//  The beneficial pattern:
//
//    input1    input2 ...                                           input3
//       |         |                                                    |
//    Reorder  Reorder ...input3                   input1  input2 ...Reorder
//           \     |     /                            \     |     /
//               Concat                 ==>               Concat
//                 |                                        |
//               Reorder                                  output
//                 |
//               output
//
//  It's worth to swap parent Reorder and Concat if the child Reorder can be eliminated.
//

bool isBeneficialToSwapConcatReorders(IE::ConcatOp& origConcatOp) {
    const auto outputConcat = origConcatOp.output();
    if (!outputConcat.hasOneUse()) {
        return false;
    }

    auto maybeReorder = mlir::dyn_cast_or_null<IE::ReorderOp>(*outputConcat.getUsers().begin());
    if (maybeReorder == nullptr) {
        return false;
    }

    size_t nonReorderInput = 0;
    size_t reorderInput = 0;

    for (const auto& arg : origConcatOp.inputs()) {
        if (auto argReorderOp = arg.getDefiningOp<IE::ReorderOp>()) {
            auto op = argReorderOp.input().getDefiningOp();
            if (mlir::isa_and_nonnull<Const::DeclareOp, IE::ReadValueOp>(op) || op == nullptr) {
                return false;
            }

            if (DimsOrder::fromValue(argReorderOp.input()) != DimsOrder::fromValue(maybeReorder.output())) {
                return false;
            }

            reorderInput++;
        } else {
            nonReorderInput++;
        }
    }

    return nonReorderInput <= reorderInput;
}

class ReorderWithConcat final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    ReorderWithConcat(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
        setDebugName("ReorderWithConcat");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origConcatOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithConcat::matchAndRewrite(IE::ConcatOp origConcatOp,
                                                       mlir::PatternRewriter& rewriter) const {
    SmallVector<mlir::Value> initialInputs;
    initialInputs.reserve(origConcatOp.inputs().size());
    SmallVector<size_t> indexNonReorder;

    Optional<DimsOrder> initialOrder;
    const bool isBeneficial = isBeneficialToSwapConcatReorders(origConcatOp);

    for (const auto& it : origConcatOp.inputs() | indexed) {
        auto arg = it.value();
        auto argReorderOp = arg.getDefiningOp<IE::ReorderOp>();
        if (argReorderOp == nullptr) {
            indexNonReorder.push_back(it.index());
            if (auto constOp = arg.getDefiningOp<Const::DeclareOp>()) {
                initialInputs.push_back(constOp.getOutput());
                continue;
            }

            if (auto readValueOp = arg.getDefiningOp<IE::ReadValueOp>()) {
                initialInputs.push_back(readValueOp.output());
                continue;
            }

            if (isBeneficial) {
                _log.trace("Got beneficial Concat: {0}", origConcatOp.getLoc());
                initialInputs.push_back(arg);
                continue;
            }

            return mlir::failure();
        }

        const auto argOrder = DimsOrder::fromValue(argReorderOp.input());
        if (!initialOrder.has_value()) {
            initialOrder = argOrder;
        } else if (argOrder != initialOrder.value()) {
            return mlir::failure();
        }

        initialInputs.push_back(argReorderOp.input());
    }

    if (!initialOrder.has_value()) {
        return mlir::failure();
    }

    const auto newConcatOrder = initialOrder.value();
    const auto originalConcatOrder = DimsOrder::fromValue(origConcatOp.output());

    // Insert reorders for ConstOps and ReadValueOps */
    for (size_t ind = 0; ind < indexNonReorder.size(); ++ind) {
        const auto index = indexNonReorder[ind];

        if (initialInputs[index].dyn_cast_or_null<mlir::BlockArgument>()) {
            initialInputs[index] =
                    insertReorderForBlockArg(initialInputs[index], newConcatOrder, rewriter, origConcatOp.getLoc());

        } else {
            auto op = initialInputs[index].getDefiningOp();
            initialInputs[index] = insertReorderForOutput(op, op->getOpResult(0), newConcatOrder, rewriter, _log);
        }
    }

    auto newConcat = rewriter.create<IE::ConcatOp>(origConcatOp->getLoc(), initialInputs, origConcatOp.per_axisAttr(),
                                                   origConcatOp.static_offsetsAttr());

    if (isBeneficial) {
        replaceChildReorderWithNewConcat(origConcatOp, newConcat, rewriter, _log);
    } else {
        rewriter.replaceOpWithNewOp<IE::ReorderOp>(origConcatOp, origConcatOp.getType(), newConcat.output(),
                                                   originalConcatOrder.toAffineMap(origConcatOp.getContext()));
    }

    return mlir::success();
}

//
// ReorderWithQuantCast
//

class ReorderWithQuantCast final : public mlir::OpRewritePattern<IE::QuantizeCastOp> {
public:
    ReorderWithQuantCast(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::QuantizeCastOp>(ctx), _log(log) {
        setDebugName("ReorderWithQuantCast");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeCastOp origQuantCastOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithQuantCast::matchAndRewrite(IE::QuantizeCastOp origQuantCastOp,
                                                          mlir::PatternRewriter& rewriter) const {
    auto origReorderOp = origQuantCastOp.input().getDefiningOp<IE::ReorderOp>();
    if (origReorderOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got reorder at '{0}' -> quantize cast at '{1}' pair", origReorderOp->getLoc(),
               origQuantCastOp->getLoc());

    auto newQuantCastOp = rewriter.create<IE::QuantizeCastOp>(origQuantCastOp->getLoc(), origReorderOp.input(),
                                                              origQuantCastOp.dstElemTypeAttr());

    rewriter.replaceOpWithNewOp<IE::ReorderOp>(origQuantCastOp, newQuantCastOp.output(), origReorderOp.dstOrderAttr());
    return mlir::success();
}

//
// ReorderWithPermuteCast
//
//               input                                input
//                 |                                    |
//              Reorder                             PermuteCast
//                 |                                    |
//             PermuteCast            ==>            Reorder
//                 |                                    |
//              Reorder                              Reorder
//                 |                                    |
//               output                               output
//
// No benefit for case that the input is a NCE task as it could fuse mem permute
// Not swap for case that two reorder could not be eliminated and the user of nextReorder is concatOp
//

class ReorderWithPermuteCast final : public mlir::OpRewritePattern<IE::PermuteCastOp> {
public:
    ReorderWithPermuteCast(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::PermuteCastOp>(ctx), _log(log) {
        setDebugName("ReorderWithPermuteCast");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::PermuteCastOp origPermuteCastOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Infer the output logical layout of original permuteCast, which is the same as the new permuteCast
// For example, based on the permutecast output physical layout = NCDHW, and permutecast
// dstOrder: [d0, d4, d1, d2, d3], it could calculate the permutecast logical layout = NDHWC
DimsOrder inferPermuteCastOutLogicalLayout(vpux::DimsOrder origReorderDstOrder,
                                           mlir::AffineMap origPermuteCastDstOrder) {
    auto origPermuteCastOutPerm = origReorderDstOrder.toPermutation();
    const auto order = DimsOrder::fromAffineMap(origPermuteCastDstOrder);
    auto targetPermutation = order.toPermutation();
    auto sourcePermutation = order.toPermutation();

    for (auto pIt = origPermuteCastOutPerm.begin(); pIt != origPermuteCastOutPerm.end(); ++pIt) {
        auto dimPosTargetPermutation = origReorderDstOrder.dimPos(Dim(pIt->ind()));
        sourcePermutation[targetPermutation[dimPosTargetPermutation].ind()] = Dim(pIt->ind());
    }

    return DimsOrder::fromPermutation(sourcePermutation);
}

// Infer the dstOrder of the new permuteCast
// For example, If the new permutecast logical layout = NDHWC, and physical layout = NDHWC
// it could calculate the new permuteCast dstOrder: [d0, d1, d2, d3, d4]
DimsOrder inferPermuteCastDstOrder(IE::ReorderOp origReorderOp, IE::PermuteCastOp origPermuteCastOp,
                                   mlir::MLIRContext* ctx) {
    const auto origReorderDstOrder = DimsOrder::fromAffineMap(origReorderOp.dstOrder());
    const auto origPermuteCastDstOrderAttr = origPermuteCastOp.dst_orderAttr();

    auto logicalLayout = inferPermuteCastOutLogicalLayout(origReorderDstOrder, origPermuteCastDstOrderAttr.getValue());

    // Calculate the permutation of new permuteCast
    const auto outputOrder = vpux::DimsOrder::fromValue(origReorderOp.input());
    const auto memPerm = vpux::getPermutationFromOrders(logicalLayout, outputOrder, ctx);

    return DimsOrder::fromAffineMap(memPerm);
}

mlir::LogicalResult ReorderWithPermuteCast::matchAndRewrite(IE::PermuteCastOp origPermuteCastOp,
                                                            mlir::PatternRewriter& rewriter) const {
    auto origReorderOp = origPermuteCastOp.input().getDefiningOp<IE::ReorderOp>();
    if (origReorderOp == nullptr) {
        return mlir::failure();
    }

    if (!origReorderOp.getResult().hasOneUse() || !origPermuteCastOp.getResult().hasOneUse()) {
        return mlir::failure();
    }

    auto nextReorderOp = mlir::dyn_cast<IE::ReorderOp>(*origPermuteCastOp.getResult().getUsers().begin());
    if (nextReorderOp == nullptr) {
        return mlir::failure();
    }

    // If Reorder is Trivial Permute, will not swap
    if (isTrivialReorder(origReorderOp)) {
        return mlir::failure();
    }

    const auto origOutOrder = DimsOrder::fromValue(origPermuteCastOp.output());
    const auto numDims = checked_cast<unsigned>(origOutOrder.numDims());
    const auto identityMap = mlir::AffineMap::getMinorIdentityMap(numDims, numDims, rewriter.getContext());
    if (origPermuteCastOp.mem_perm() != identityMap) {
        return mlir::failure();
    }

    // No benefit for case that NCE tasks could fuse mem permute
    auto layerWithPermute = origReorderOp.input().getDefiningOp<IE::LayerWithPermuteInterface>();
    if (layerWithPermute != nullptr && layerWithPermute.isSupportedPermutation(origReorderOp)) {
        return mlir::failure();
    }

    // Infer the DstOrder of the new permuteCast
    auto newDstOrder = inferPermuteCastDstOrder(origReorderOp, origPermuteCastOp, rewriter.getContext());
    const auto newDstOrderAttr = mlir::AffineMapAttr::get(newDstOrder.toAffineMap(rewriter.getContext()));
    auto newMemPermAttr = origPermuteCastOp.mem_permAttr();

    // Not swap in case that the two reorder could not be eliminated and the nextReorder's user is concatOp
    const auto nextReorderOutOrder = DimsOrder::fromAffineMap(nextReorderOp.dstOrder());
    bool canEliminate = newDstOrder == nextReorderOutOrder;
    for (auto* nextReorderUser : llvm::make_early_inc_range(nextReorderOp.getResult().getUsers())) {
        auto nextConcat = mlir::dyn_cast<IE::ConcatOp>(nextReorderUser);
        if (nextConcat != nullptr && !canEliminate) {
            return mlir::failure();
        }
    }

    _log.trace("Got reorder at '{0}' -> permute cast at '{1}' pair", origReorderOp->getLoc(),
               origPermuteCastOp->getLoc());

    auto newPermuteCastOp = rewriter.create<IE::PermuteCastOp>(origPermuteCastOp->getLoc(), origReorderOp.input(),
                                                               newDstOrderAttr, newMemPermAttr);

    rewriter.replaceOpWithNewOp<IE::ReorderOp>(origPermuteCastOp, newPermuteCastOp.output(),
                                               origPermuteCastOp.dst_orderAttr());
    return mlir::success();
}

//
// ReorderWithConvert
//

class ReorderWithConvert final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    ReorderWithConvert(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvertOp>(ctx), _log(log) {
        setDebugName("ReorderWithConvert");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvertOp convertOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithConvert::matchAndRewrite(IE::ConvertOp convertOp,
                                                        mlir::PatternRewriter& rewriter) const {
    // Note that in this case we replace Convert -> Reorder with Reorder -> Convert
    // This is an opposite behavior, compared to other rewriters
    if (!convertOp.getResult().hasOneUse()) {
        return matchFailed(_log.nest(), rewriter, convertOp, "ConvertOp has more then one user");
    }

    auto origReorderOp = mlir::dyn_cast<IE::ReorderOp>(*convertOp.getResult().getUsers().begin());
    if (origReorderOp == nullptr) {
        return mlir::failure();
    }

    const auto srcType = convertOp.input().getType();
    const auto dstElemType = convertOp.dstElemType();
    if (getElemTypeSize(srcType) >= getElemTypeSize(dstElemType)) {
        return matchFailed(rewriter, convertOp, "Convert doesn't increase data size");
    }

    auto newReorderOp =
            rewriter.create<IE::ReorderOp>(origReorderOp->getLoc(), convertOp.input(), origReorderOp.dstOrderAttr());

    rewriter.replaceOpWithNewOp<IE::ConvertOp>(origReorderOp, origReorderOp.getType(), newReorderOp.output(),
                                               convertOp.dstElemTypeAttr());

    return mlir::success();
}

// Search op on the consumer chain(bypass view like operations), until target operation is found or reach the last
// consumer.
// Return mlir::Operation if target op is found, otherwise return mlir::failure().
mlir::FailureOr<mlir::Operation*> searchOpConsumers(mlir::Operation* op,
                                                    const std::function<bool(mlir::Operation*)>& isTargetOpFound) {
    if (op == nullptr || !op->hasOneUse()) {
        return mlir::failure();
    }

    mlir::Operation* operation = op;
    while (operation && !operation->getUsers().empty()) {
        auto user = *(operation->getUsers().begin());

        if (isTargetOpFound(user)) {
            return user;
        } else if (IE::isPureViewOp(user)) {
            if (!user->hasOneUse()) {
                return mlir::failure();
            }
            operation = user;
            continue;
        } else {
            break;
        }
    }
    return mlir::failure();
}

mlir::FailureOr<mlir::Operation*> getConvertOrReturnOpConsumer(mlir::Operation* op) {
    std::function<bool(mlir::Operation*)> isConvertOrReturnOpFound = [](mlir::Operation* op) -> bool {
        return mlir::isa<IE::ConvertOp, mlir::func::ReturnOp>(op);
    };

    return searchOpConsumers(op, isConvertOrReturnOpFound);
}

mlir::FailureOr<mlir::Operation*> getReturnOpConsumer(mlir::Operation* op) {
    std::function<bool(mlir::Operation*)> isReturnOpFound = [](mlir::Operation* op) -> bool {
        return mlir::isa<mlir::func::ReturnOp>(op);
    };

    return searchOpConsumers(op, isReturnOpFound);
}

bool doesConvertOpIncreaseElemSize(IE::ConvertOp convertOp) {
    const auto srcType = convertOp.input().getType();
    const auto dstElemType = convertOp.dstElemType();
    return getElemTypeSize(srcType) <= getElemTypeSize(dstElemType);
}

//
// ReorderWithLayer
//

class ReorderWithLayer final : public mlir::OpInterfaceRewritePattern<IE::LayoutInfoOpInterface> {
public:
    ReorderWithLayer(mlir::MLIRContext* ctx, Logger log, const bool seOpsEnabled)
            : mlir::OpInterfaceRewritePattern<IE::LayoutInfoOpInterface>(ctx), _log(log), _seOpsEnabled(seOpsEnabled) {
        setDebugName("ReorderWithLayer");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LayoutInfoOpInterface layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    bool _seOpsEnabled;
};

mlir::LogicalResult ReorderWithLayer::matchAndRewrite(IE::LayoutInfoOpInterface layerOp,
                                                      mlir::PatternRewriter& rewriter) const {
    if (mlir::isa<IE::ReorderOp>(layerOp)) {
        return mlir::failure();
    }

    _log.trace("Got layer operation '{0}' at '{1}'", layerOp->getName(), layerOp->getLoc());

    auto argReorderOp = layerOp->getOperand(0).getDefiningOp<IE::ReorderOp>();
    if (argReorderOp == nullptr) {
        return mlir::failure();
    }

    // Skip reorder propagation for below cases:
    // 1.ReorderOp's consumers are pure view like ops with ReturnOp
    //   e.g.Reorder->AffineReshape->PermuteCast->QuantizeCast->ReturnOp
    // 2.ReorderOp's consumers are pure view like ops with ConvertOp and ReturnOp.
    //   Additionally ConvertOp is increasing element type size.
    //   e.g.Reorder->AffineReshape->ConvertOp(F16->F32)->AffineReshape->PermuteCast->QuantizeCast->ReturnOp
    auto getConsumerResult = getConvertOrReturnOpConsumer(argReorderOp);
    if (!mlir::failed(getConsumerResult)) {
        auto convertOrReturnOp = getConsumerResult.value();
        if (mlir::isa<mlir::func::ReturnOp>(convertOrReturnOp)) {
            return mlir::failure();
        } else if (mlir::isa<IE::ConvertOp>(convertOrReturnOp)) {
            auto convertOp = mlir::dyn_cast<IE::ConvertOp>(convertOrReturnOp);
            bool convertOpHasReturnConsumer = !mlir::failed(getReturnOpConsumer(convertOp));
            if (convertOpHasReturnConsumer && doesConvertOpIncreaseElemSize(convertOp)) {
                return mlir::failure();
            }
        } else {
            VPUX_THROW("Unexpected operation '{0}' at '{1}'", convertOrReturnOp->getName(),
                       convertOrReturnOp->getLoc());
        }
    }

    const auto propagatingOrder = DimsOrder::fromValue(argReorderOp.input());

    // Propagate first input layout and infer layout info
    auto orderInfo = layerOp.getLayoutInfo();
    orderInfo.setInput(0, propagatingOrder);
    layerOp.inferLayoutInfo(orderInfo, _seOpsEnabled);
    if (orderInfo.getInput(0) != propagatingOrder) {
        return matchFailed(_log.nest(), rewriter, layerOp, "Layer doesn't support propagating order {0}",
                           propagatingOrder);
    }

    // Check if additional reorders for other inputs are needed
    for (auto ind : irange<size_t>(1, orderInfo.getNumInputs())) {
        const auto input = layerOp->getOperand(checked_cast<uint32_t>(ind));
        const auto order = DimsOrder::fromValue(input);
        const auto isConstInput = mlir::isa_and_nonnull<Const::DeclareOp>(input.getDefiningOp());
        const auto isReorderInput = mlir::isa_and_nonnull<IE::ReorderOp>(input.getDefiningOp());

        if (order != orderInfo.getInput(ind) && !isConstInput && !isReorderInput) {
            return matchFailed(_log.nest(), rewriter, layerOp, "Non-constant inputs require additional Reorders");
        }
    }

    rewriter.startRootUpdate(layerOp);

    _log.nest(1).trace("Remove Reorder before the first input");
    layerOp->getOpOperand(0).set(argReorderOp.input());

    const auto inputs = layerOp->getOpOperands();
    for (auto i : irange<size_t>(1, inputs.size())) {
        auto& input = inputs[i];

        const auto curOrder = DimsOrder::fromValue(input.get());
        const auto supportedOrder = orderInfo.getInput(i);

        _log.nest(1).trace("Process input #{0}", i);
        if (curOrder != supportedOrder) {
            insertReorderForInput(layerOp, input, supportedOrder, rewriter, _log.nest());
        }
    }

    const auto outputs = layerOp->getOpResults();
    for (auto i : irange(outputs.size())) {
        auto output = outputs[i];

        const auto curOrder = DimsOrder::fromValue(output);
        const auto supportedOrder = orderInfo.getOutput(i);

        _log.nest(1).trace("Process output #{0}", i);
        if (curOrder != supportedOrder) {
            changeDimsOrder(output, supportedOrder, _log.nest());
            insertReorderForOutput(layerOp, output, curOrder, rewriter, _log.nest());
        }
    }

    rewriter.finalizeRootUpdate(layerOp);

    return mlir::success();
}

//
// ReorderWithAssign
//

class ReorderWithAssign final : public mlir::OpRewritePattern<IE::AssignOp> {
public:
    ReorderWithAssign(mlir::MLIRContext* ctx, const mlir::DenseSet<llvm::StringRef> inputSet, Logger log)
            : mlir::OpRewritePattern<IE::AssignOp>(ctx), _assignNameSetToOptimize(inputSet), _log(log) {
        setDebugName("ReorderWithAssign");
    }

    mlir::LogicalResult matchAndRewrite(IE::AssignOp origAssignOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::DenseSet<llvm::StringRef> _assignNameSetToOptimize;
    Logger _log;
};

// Remove reorder connected to assign op, and change the layout of assign op
mlir::LogicalResult ReorderWithAssign::matchAndRewrite(IE::AssignOp origAssignOp,
                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got ReorderWithAssign layer at '{1}'", getDebugName(), origAssignOp->getLoc());

    if (!_assignNameSetToOptimize.count(origAssignOp.name())) {
        return mlir::failure();
    }

    auto prevOp = origAssignOp.input().getDefiningOp();

    if (!prevOp->getResult(0).hasOneUse()) {
        return matchFailed(_log.nest(), rewriter, prevOp, "prev ReorderOp has more then one user");
    }

    if (!mlir::isa_and_nonnull<IE::ReorderOp>(prevOp)) {
        return mlir::failure();
    }

    auto prevOpInputDimsOrder = DimsOrder::fromValue(prevOp->getOperand(0));
    vpux::changeDimsOrder(origAssignOp.input(), prevOpInputDimsOrder, _log);
    vpux::changeDimsOrder(origAssignOp.output(), prevOpInputDimsOrder, _log);
    rewriter.replaceOp(prevOp, prevOp->getOperand(0));
    return mlir::success();
}

//
// ReorderWithReadValue
//

class ReorderWithReadValue final : public mlir::OpRewritePattern<IE::ReadValueOp> {
public:
    ReorderWithReadValue(mlir::MLIRContext* ctx, const mlir::DenseSet<llvm::StringRef> inputSet, Logger log)
            : mlir::OpRewritePattern<IE::ReadValueOp>(ctx), _readValueNameSetToOptimize(inputSet), _log(log) {
        setDebugName("ReorderWithReadValue");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReadValueOp origReadValueOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::DenseSet<llvm::StringRef> _readValueNameSetToOptimize;
    Logger _log;
};

// Revert the order between read value and reorder to remove reorders connected to read value op
mlir::LogicalResult ReorderWithReadValue::matchAndRewrite(IE::ReadValueOp origReadValueOp,
                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got ReorderWithReadValue layer at '{1}'", getDebugName(), origReadValueOp->getLoc());

    if (!_readValueNameSetToOptimize.count(origReadValueOp.name())) {
        return mlir::failure();
    }

    if (!origReadValueOp.output().hasOneUse()) {
        return matchFailed(_log.nest(), rewriter, origReadValueOp, "ReadValue has more then one user");
    }

    auto origReorderOp = mlir::dyn_cast<IE::ReorderOp>(*origReadValueOp.output().getUsers().begin());

    if (origReorderOp == nullptr) {
        return mlir::failure();
    }

    // Note that in this case we replace Declare -> ReadValue -> Reorder with Declare -> Reorder -> ReadValue,
    // Then Declare -> Reorder -> ReadValue will be changed to Declare -> ReadValue
    auto newReorderOp = rewriter.create<IE::ReorderOp>(origReorderOp->getLoc(), origReadValueOp.input(),
                                                       origReorderOp.dstOrderAttr());

    rewriter.replaceOpWithNewOp<IE::ReadValueOp>(origReorderOp, newReorderOp.output(), origReadValueOp.name());
    // erase readvalue ops which has no more nodes next
    rewriter.eraseOp(origReadValueOp);

    return mlir::success();
}

//
// OptimizeReordersPass
//

class OptimizeReordersPass final : public IE::OptimizeReordersBase<OptimizeReordersPass> {
public:
    explicit OptimizeReordersPass(const bool seOpsEnabled, Logger log): _seOpsEnabled(seOpsEnabled) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

    mlir::DenseSet<llvm::StringRef> getReadValueAndAssignPairs(mlir::func::FuncOp func, Logger log);

private:
    bool _seOpsEnabled;
};

mlir::DenseSet<llvm::StringRef> OptimizeReordersPass::getReadValueAndAssignPairs(mlir::func::FuncOp func, Logger log) {
    // traverse to get all read value and assign ops, and record them with reorders connected to
    mlir::DenseMap<llvm::StringRef, DimsOrder> readValueMap;
    mlir::DenseMap<llvm::StringRef, DimsOrder> assignMap;

    func->walk([&](IE::ReadValueOp readValueOp) {
        auto nextOp = *readValueOp.output().getUsers().begin();
        if (mlir::isa_and_nonnull<IE::ReorderOp>(nextOp) && readValueOp.getResult().hasOneUse()) {
            // Only if readValue has one user, i.e. Reorder Op, the convert is legal to happen
            log.trace("Found Read Value Operation with Reorder Op'{0}' ", readValueOp->getLoc());
            auto nextOpResultDimsOrder = DimsOrder::fromValue(nextOp->getResult(0));
            readValueMap.insert({readValueOp.name(), nextOpResultDimsOrder});
        }
    });

    func->walk([&](IE::AssignOp assignOp) {
        auto prevOp = assignOp.input().getDefiningOp();
        if (mlir::isa_and_nonnull<IE::ReorderOp>(prevOp) && prevOp->getResult(0).hasOneUse()) {
            // Only if prev Reorder Op has one user, i.e. Assign Op, the convert is legal to happen
            log.trace("Found Assign Operation with Reorder Op'{0}' ", assignOp->getLoc());
            auto prevOpInputDimsOrder = DimsOrder::fromValue(prevOp->getOperand(0));
            assignMap.insert({assignOp.name(), prevOpInputDimsOrder});
        }
    });

    mlir::DenseSet<llvm::StringRef> readValueAndAssignCommonPairs;
    for (auto ex : readValueMap) {
        if (assignMap[ex.first] == ex.second) {
            readValueAndAssignCommonPairs.insert(ex.first);
        }
    }

    return readValueAndAssignCommonPairs;
}

void OptimizeReordersPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ReorderWithShapeChange<IE::ReshapeOp>>(&ctx, _log);
    patterns.add<ReorderWithShapeChange<IE::ShapeCastOp>>(&ctx, _log);
    patterns.add<ReorderWithSubView>(&ctx, _log);
    patterns.add<ReorderWithExpand>(&ctx, _log);
    patterns.add<ReorderWithSplit>(&ctx, _log);
    patterns.add<ReorderWithConcat>(&ctx, _log);
    patterns.add<ReorderWithQuantCast>(&ctx, _log);
    patterns.add<ReorderWithLayer>(&ctx, _log, _seOpsEnabled);
    patterns.add<ReorderWithPermuteCast>(&ctx, _log);
    IE::ReorderOp::getCanonicalizationPatterns(patterns, &ctx);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
        return;
    }

    auto readValueAndAssignCommonPairs = getReadValueAndAssignPairs(func, _log);
    mlir::RewritePatternSet rvaPatterns(&ctx);
    rvaPatterns.add<ReorderWithAssign>(&ctx, readValueAndAssignCommonPairs, _log);
    rvaPatterns.add<ReorderWithReadValue>(&ctx, readValueAndAssignCommonPairs, _log);
    IE::ReorderOp::getCanonicalizationPatterns(rvaPatterns, &ctx);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(rvaPatterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }

    mlir::RewritePatternSet cleanupPatterns(&ctx);
    cleanupPatterns.add<ReorderWithConvert>(&ctx, _log);
    IE::ReorderOp::getCanonicalizationPatterns(cleanupPatterns, &ctx);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(cleanupPatterns),
                                                        getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeReordersPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeReordersPass(const bool seOpsEnabled, Logger log) {
    return std::make_unique<OptimizeReordersPass>(seOpsEnabled, log);
}
