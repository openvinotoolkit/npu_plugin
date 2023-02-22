//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/utils/permute_infer.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/TypeSwitch.h>
#include <vpux/compiler/conversion.hpp>

using namespace vpux;

namespace {

Dim getHighestDim(ShapeRef shape, const DimsOrder& dimOrder) {
    auto highestDim = Dim(0);
    for (auto idx : irange(dimOrder.numDims())) {
        auto curDim = dimOrder.dimAt(idx);
        if (shape[curDim] != 1) {
            highestDim = curDim;
            break;
        }
    }
    return highestDim;
}

bool doesSliceAndPermutationModifySameAxis(DimsOrder perm, ArrayRef<uint64_t> sliceAxes, Logger log) {
    auto order = to_small_vector(irange(perm.numDims()) | transformed([&](uint64_t idx) {
                                     return checked_cast<uint64_t>(perm.dimAt(idx).ind());
                                 }));

    for (auto axis : sliceAxes) {
        if (order[axis] != axis) {
            log.trace("Layer modifies sliced axis: {0}", axis);
            return true;
        }
    }

    return false;
}

template <typename ConcreteOp>
class MoveLayerBeforeSlice : public mlir::OpRewritePattern<ConcreteOp> {
public:
    MoveLayerBeforeSlice(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    virtual bool doesSliceAndLayerOpModifySameAxis(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                                   ConcreteOp layerOp) const = 0;
    virtual bool sameAttributes(ConcreteOp layerOp, ConcreteOp currLayerOp) const = 0;
    virtual mlir::Operation* createNewLayerOp(ConcreteOp layerOp, IE::SliceOp sliceOp,
                                              mlir::PatternRewriter& rewriter) const;
    virtual SmallVector<int64_t> getNewSizes(IE::SliceOp sliceOp, ConcreteOp layerOp) const = 0;
    virtual SmallVector<int64_t> getNewOffsets(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                               ConcreteOp layerOp) const;

protected:
    Logger _log;
};

template <typename ConcreteOp>
mlir::LogicalResult MoveLayerBeforeSlice<ConcreteOp>::matchAndRewrite(ConcreteOp layerOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    auto maybeSliceOp = layerOp.input().template getDefiningOp<IE::SliceOp>();
    if (maybeSliceOp == nullptr) {
        return mlir::failure();
    }

    auto sliceSrcUsers = maybeSliceOp.source().getUsers();

    auto hasAnotherSlice = llvm::any_of(sliceSrcUsers, [&](mlir::Operation* user) {
        auto anotherSlice = mlir::dyn_cast<IE::SliceOp>(user);
        return anotherSlice != nullptr && anotherSlice != maybeSliceOp;
    });

    if (!hasAnotherSlice) {
        return mlir::failure();
    }

    _log.trace("Got layer op: {0}", layerOp);
    _log.trace("Parent slice: {0}", maybeSliceOp);

    auto sliceInShape = getShape(maybeSliceOp.source());
    auto sizes = parseIntArrayAttr<int64_t>(maybeSliceOp.static_sizes());

    SmallVector<uint64_t> sliceAxes;
    for (size_t dimIdx = 0; dimIdx < sizes.size(); dimIdx++) {
        if (sliceInShape[Dim(dimIdx)] != sizes[dimIdx]) {
            sliceAxes.push_back(static_cast<uint64_t>(dimIdx));
        }
    }

    if (doesSliceAndLayerOpModifySameAxis(maybeSliceOp, sliceAxes, layerOp)) {
        return mlir::failure();
    }

    const auto isSameSliceLayerBranch = [&](mlir::Operation* user) {
        auto currSliceOp = mlir::dyn_cast<IE::SliceOp>(user);
        if (currSliceOp == nullptr) {
            return true;
        }

        if (!currSliceOp.result().hasOneUse()) {
            return false;
        }

        if (currSliceOp.static_sizesAttr() != maybeSliceOp.static_sizesAttr()) {
            return false;
        }

        auto currLayerOp = mlir::dyn_cast<ConcreteOp>(*currSliceOp.result().getUsers().begin());
        if (currLayerOp == nullptr) {
            return false;
        }

        return sameAttributes(layerOp, currLayerOp);
    };

    for (auto user : maybeSliceOp.source().getUsers()) {
        if (!isSameSliceLayerBranch(user)) {
            return mlir::failure();
        }
    }

    auto newLayerOp = createNewLayerOp(layerOp, maybeSliceOp, rewriter);
    _log.trace("Create new layer op: {0}", newLayerOp);

    auto newSizes = getNewSizes(maybeSliceOp, layerOp);
    for (auto user : maybeSliceOp.source().getUsers()) {
        auto sliceOp = mlir::dyn_cast<IE::SliceOp>(user);
        if (sliceOp == nullptr) {
            continue;
        }

        auto currLayerOp = mlir::dyn_cast<ConcreteOp>(*sliceOp.result().getUsers().begin());
        auto newOffsets = getNewOffsets(sliceOp, sliceAxes, currLayerOp);

        auto newSlice = rewriter.create<IE::SliceOp>(currLayerOp->getLoc(), newLayerOp->getResult(0),
                                                     getIntArrayAttr(newLayerOp->getContext(), newOffsets),
                                                     getIntArrayAttr(newLayerOp->getContext(), newSizes));
        _log.trace("Create new Slice: {0}", newSlice);

        rewriter.replaceOp(currLayerOp, newSlice.result());
    }

    return mlir::success();
}

template <typename ConcreteOp>
mlir::Operation* MoveLayerBeforeSlice<ConcreteOp>::createNewLayerOp(ConcreteOp layerOp, IE::SliceOp sliceOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    mlir::BlockAndValueMapping mapper;
    mapper.map(layerOp->getOperands(), makeArrayRef({sliceOp.source()}));

    rewriter.setInsertionPointAfterValue(sliceOp.source());
    auto* newLayerOp = rewriter.clone(*layerOp.getOperation(), mapper);
    vpux::inferReturnTypes(newLayerOp, vpux::InferShapedTypeMode::ALL);
    return newLayerOp;
}

template <typename ConcreteOp>
SmallVector<int64_t> MoveLayerBeforeSlice<ConcreteOp>::getNewOffsets(IE::SliceOp sliceOp, ArrayRef<uint64_t>,
                                                                     ConcreteOp) const {
    return parseIntArrayAttr<int64_t>(sliceOp.static_offsets());
}

//
// MoveExpandBeforeSlice
//

class MoveExpandBeforeSlice final : public MoveLayerBeforeSlice<IE::ExpandOp> {
public:
    MoveExpandBeforeSlice(mlir::MLIRContext* ctx, Logger log): MoveLayerBeforeSlice<IE::ExpandOp>(ctx, log) {
    }

private:
    bool doesSliceAndLayerOpModifySameAxis(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                           IE::ExpandOp layerOp) const override;
    bool sameAttributes(IE::ExpandOp layerOp, IE::ExpandOp currLayerOp) const override;
    SmallVector<int64_t> getNewSizes(IE::SliceOp sliceOp, IE::ExpandOp layerOp) const override;
};

bool MoveExpandBeforeSlice::doesSliceAndLayerOpModifySameAxis(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                                              IE::ExpandOp layerOp) const {
    auto sizes = parseIntArrayAttr<int64_t>(sliceOp.static_sizes());
    auto expandedShape = getShape(layerOp.output());

    for (auto axis : sliceAxes) {
        if (sizes[axis] != expandedShape[Dim(axis)]) {
            _log.trace("Expand modifies sliced axis: {0}", axis);
            return true;
        }
    }

    return false;
}

bool MoveExpandBeforeSlice::sameAttributes(IE::ExpandOp layerOp, IE::ExpandOp currLayerOp) const {
    return layerOp.pads_beginAttr() == currLayerOp.pads_beginAttr() &&
           layerOp.pads_endAttr() == currLayerOp.pads_endAttr();
}

SmallVector<int64_t> MoveExpandBeforeSlice::getNewSizes(IE::SliceOp sliceOp, IE::ExpandOp layerOp) const {
    auto begin = parseIntArrayAttr<int64_t>(layerOp.pads_begin());
    auto end = parseIntArrayAttr<int64_t>(layerOp.pads_end());
    auto expandedShape = getShape(layerOp.output());

    auto sizes = parseIntArrayAttr<int64_t>(sliceOp.static_sizes());

    SmallVector<int64_t> newSizes(expandedShape.size(), 0);
    for (size_t i = 0; i < expandedShape.size(); i++) {
        newSizes[i] = begin[i] + sizes[i] + end[i];
    }

    return newSizes;
}

//
// MoveReorderBeforeSlice
//

class MoveReorderBeforeSlice final : public MoveLayerBeforeSlice<IE::ReorderOp> {
public:
    MoveReorderBeforeSlice(mlir::MLIRContext* ctx, Logger log): MoveLayerBeforeSlice<IE::ReorderOp>(ctx, log) {
    }

public:
    bool doesSliceAndLayerOpModifySameAxis(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                           IE::ReorderOp layerOp) const override;
    bool sameAttributes(IE::ReorderOp layerOp, IE::ReorderOp currLayerOp) const override;
    SmallVector<int64_t> getNewSizes(IE::SliceOp sliceOp, IE::ReorderOp layerOp) const override;
};

bool MoveReorderBeforeSlice::doesSliceAndLayerOpModifySameAxis(IE::SliceOp, ArrayRef<uint64_t> sliceAxes,
                                                               IE::ReorderOp layerOp) const {
    const auto perm = DimsOrder::fromAffineMap(layerOp.dstOrder());
    return doesSliceAndPermutationModifySameAxis(perm, sliceAxes, _log);
}
bool MoveReorderBeforeSlice::sameAttributes(IE::ReorderOp layerOp, IE::ReorderOp currLayerOp) const {
    return layerOp.dstOrder() == currLayerOp.dstOrder();
}

SmallVector<int64_t> MoveReorderBeforeSlice::getNewSizes(IE::SliceOp sliceOp, IE::ReorderOp) const {
    return parseIntArrayAttr<int64_t>(sliceOp.static_sizes());
}

//
// MoveTransposeBeforeSlice
//

class MoveTransposeBeforeSlice final : public MoveLayerBeforeSlice<IE::TransposeOp> {
public:
    MoveTransposeBeforeSlice(mlir::MLIRContext* ctx, Logger log): MoveLayerBeforeSlice<IE::TransposeOp>(ctx, log) {
    }

public:
    bool doesSliceAndLayerOpModifySameAxis(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                           IE::TransposeOp layerOp) const override;
    bool sameAttributes(IE::TransposeOp layerOp, IE::TransposeOp currLayerOp) const override;
    SmallVector<int64_t> getNewSizes(IE::SliceOp sliceOp, IE::TransposeOp layerOp) const override;
};

bool MoveTransposeBeforeSlice::doesSliceAndLayerOpModifySameAxis(IE::SliceOp, ArrayRef<uint64_t> sliceAxes,
                                                                 IE::TransposeOp layerOp) const {
    const auto perm = DimsOrder::fromAffineMap(layerOp.order_value().getValue());
    return doesSliceAndPermutationModifySameAxis(perm, sliceAxes, _log);
}

bool MoveTransposeBeforeSlice::sameAttributes(IE::TransposeOp layerOp, IE::TransposeOp currLayerOp) const {
    return layerOp.order_value() == currLayerOp.order_value();
}

SmallVector<int64_t> MoveTransposeBeforeSlice::getNewSizes(IE::SliceOp sliceOp, IE::TransposeOp layerOp) const {
    const auto perm = DimsOrder::fromAffineMap(layerOp.order_value().getValue());
    auto originOrder = to_small_vector(irange(perm.numDims()) | transformed([&](uint64_t idx) {
                                           return checked_cast<uint64_t>(perm.dimAt(idx).ind());
                                       }));

    auto sizes = parseIntArrayAttr<int64_t>(sliceOp.static_sizes());
    SmallVector<int64_t> newSizes(originOrder.size(), 0);
    for (size_t i = 0; i < originOrder.size(); i++) {
        newSizes[i] = sizes[originOrder[i]];
    }

    return newSizes;
}

//
// MovePermuteCastBeforeSlice
//

class MovePermuteCastBeforeSlice final : public MoveLayerBeforeSlice<IE::PermuteCastOp> {
public:
    MovePermuteCastBeforeSlice(mlir::MLIRContext* ctx, Logger log): MoveLayerBeforeSlice<IE::PermuteCastOp>(ctx, log) {
    }

public:
    bool doesSliceAndLayerOpModifySameAxis(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                           IE::PermuteCastOp layerOp) const override;
    bool sameAttributes(IE::PermuteCastOp layerOp, IE::PermuteCastOp currLayerOp) const override;
    SmallVector<int64_t> getNewSizes(IE::SliceOp sliceOp, IE::PermuteCastOp layerOp) const override;
};

bool MovePermuteCastBeforeSlice::doesSliceAndLayerOpModifySameAxis(IE::SliceOp, ArrayRef<uint64_t> sliceAxes,
                                                                   IE::PermuteCastOp layerOp) const {
    const auto orderPerm = DimsOrder::fromAffineMap(layerOp.dst_order());
    const auto memPerm = DimsOrder::fromAffineMap(layerOp.mem_perm());

    return doesSliceAndPermutationModifySameAxis(orderPerm, sliceAxes, _log) ||
           doesSliceAndPermutationModifySameAxis(memPerm, sliceAxes, _log);
}

bool MovePermuteCastBeforeSlice::sameAttributes(IE::PermuteCastOp layerOp, IE::PermuteCastOp currLayerOp) const {
    return layerOp.dst_order() == currLayerOp.dst_order() && layerOp.mem_perm() == currLayerOp.mem_perm();
}

SmallVector<int64_t> MovePermuteCastBeforeSlice::getNewSizes(IE::SliceOp, IE::PermuteCastOp layerOp) const {
    mlir::SmallVector<mlir::ShapedTypeComponents> inferredReturnShapes;

    inferPermuteReturnTypeComponents(layerOp.input(), layerOp.mem_perm(), layerOp.dst_order(), inferredReturnShapes,
                                     false);

    VPUX_THROW_WHEN(inferredReturnShapes.size() != 1, "Should should be 1 but got {0}", inferredReturnShapes.size());

    auto dims = inferredReturnShapes.front().getDims();

    return mlir::SmallVector<int64_t>{dims.begin(), dims.end()};
}

//
// MoveAffineReshapeBeforeSlice
//

class MoveAffineReshapeBeforeSlice final : public MoveLayerBeforeSlice<IE::AffineReshapeOp> {
public:
    MoveAffineReshapeBeforeSlice(mlir::MLIRContext* ctx, Logger log)
            : MoveLayerBeforeSlice<IE::AffineReshapeOp>(ctx, log) {
    }

public:
    bool doesSliceAndLayerOpModifySameAxis(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                           IE::AffineReshapeOp layerOp) const override;
    bool sameAttributes(IE::AffineReshapeOp layerOp, IE::AffineReshapeOp currLayerOp) const override;
    SmallVector<int64_t> getNewSizes(IE::SliceOp sliceOp, IE::AffineReshapeOp layerOp) const override;
    SmallVector<int64_t> getNewOffsets(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                       IE::AffineReshapeOp layerOp) const override;

    mlir::Operation* createNewLayerOp(IE::AffineReshapeOp layerOp, IE::SliceOp sliceOp,
                                      mlir::PatternRewriter& rewriter) const override;
};

bool MoveAffineReshapeBeforeSlice::doesSliceAndLayerOpModifySameAxis(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                                                     IE::AffineReshapeOp) const {
    // only move AffineReshape when slice on the highest dimension
    if (sliceAxes.size() != 1) {
        return true;
    }

    auto sliceAxis = sliceAxes[0];
    auto inType = sliceOp.source().getType().dyn_cast<vpux::NDTypeInterface>();
    auto dimOrder = inType.getDimsOrder();
    auto shape = inType.getShape();
    auto highestDim = getHighestDim(shape, dimOrder);
    return checked_cast<uint64_t>(highestDim.ind()) != sliceAxis;
}

bool MoveAffineReshapeBeforeSlice::sameAttributes(IE::AffineReshapeOp layerOp, IE::AffineReshapeOp currLayerOp) const {
    return layerOp.dim_mapping() == currLayerOp.dim_mapping() && layerOp.shape_value() == currLayerOp.shape_value();
}

SmallVector<int64_t> MoveAffineReshapeBeforeSlice::getNewSizes(IE::SliceOp, IE::AffineReshapeOp layerOp) const {
    return vpux::parseIntArrayAttr<int64_t>(layerOp.shape_value());
}

SmallVector<int64_t> MoveAffineReshapeBeforeSlice::getNewOffsets(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                                                 IE::AffineReshapeOp layerOp) const {
    auto sliceOffset = parseIntArrayAttr<int64_t>(sliceOp.static_offsets());
    auto sliceSize = parseIntArrayAttr<int64_t>(sliceOp.static_sizes());
    VPUX_THROW_UNLESS(sliceAxes.size() == 1, "Unexpected slice axes for {0}", sliceOp);
    auto sliceOffsetVal = sliceOffset[sliceAxes[0]];
    auto sliceSizeVal = sliceSize[sliceAxes[0]];
    auto sliceIdx = sliceOffsetVal / sliceSizeVal;
    auto newSizes = getNewSizes(sliceOp, layerOp);
    SmallVector<int64_t> newOffsets(newSizes.size(), 0);

    auto outType = layerOp.output().getType().dyn_cast<vpux::NDTypeInterface>();
    auto dimOrder = outType.getDimsOrder();
    auto sliceDim = dimOrder.dimAt(0);
    newOffsets[sliceDim.ind()] = sliceIdx * newSizes[sliceDim.ind()];
    return newOffsets;
}

mlir::Operation* MoveAffineReshapeBeforeSlice::createNewLayerOp(IE::AffineReshapeOp layerOp, IE::SliceOp sliceOp,
                                                                mlir::PatternRewriter& rewriter) const {
    auto outType = layerOp.output().getType().dyn_cast<vpux::NDTypeInterface>();
    auto outShape = outType.getShape();
    auto outDim = outType.getDimsOrder();
    auto sliceDim = outDim.dimAt(0);
    auto inShape = getShape(sliceOp.source());
    auto restSize = outShape.totalSize() / outShape[sliceDim];
    Shape newOutShape(outShape.raw());
    newOutShape[sliceDim] = inShape.totalSize() / restSize;
    rewriter.setInsertionPointAfterValue(sliceOp.source());
    return rewriter.create<IE::AffineReshapeOp>(layerOp.getLoc(), sliceOp.source(), layerOp.dim_mapping(),
                                                vpux::getIntArrayAttr(rewriter, to_small_vector(newOutShape)));
}

//
// MoveMemPermuteBeforeSlice
//

class MoveMemPermuteBeforeSlice final : public MoveLayerBeforeSlice<IE::MemPermuteOp> {
public:
    MoveMemPermuteBeforeSlice(mlir::MLIRContext* ctx, Logger log): MoveLayerBeforeSlice<IE::MemPermuteOp>(ctx, log) {
    }

public:
    bool doesSliceAndLayerOpModifySameAxis(IE::SliceOp sliceOp, ArrayRef<uint64_t> sliceAxes,
                                           IE::MemPermuteOp layerOp) const override;
    bool sameAttributes(IE::MemPermuteOp layerOp, IE::MemPermuteOp currLayerOp) const override;
    SmallVector<int64_t> getNewSizes(IE::SliceOp sliceOp, IE::MemPermuteOp layerOp) const override;
};

bool MoveMemPermuteBeforeSlice::doesSliceAndLayerOpModifySameAxis(IE::SliceOp, ArrayRef<uint64_t> sliceAxes,
                                                                  IE::MemPermuteOp layerOp) const {
    const auto orderPerm = DimsOrder::fromAffineMap(layerOp.dst_order());
    const auto memPerm = DimsOrder::fromAffineMap(layerOp.mem_perm());

    return doesSliceAndPermutationModifySameAxis(orderPerm, sliceAxes, _log) ||
           doesSliceAndPermutationModifySameAxis(memPerm, sliceAxes, _log);
}

bool MoveMemPermuteBeforeSlice::sameAttributes(IE::MemPermuteOp layerOp, IE::MemPermuteOp currLayerOp) const {
    return layerOp.dst_order() == currLayerOp.dst_order() && layerOp.mem_perm() == currLayerOp.mem_perm();
}

SmallVector<int64_t> MoveMemPermuteBeforeSlice::getNewSizes(IE::SliceOp, IE::MemPermuteOp layerOp) const {
    mlir::SmallVector<mlir::ShapedTypeComponents> inferredReturnShapes;

    inferPermuteReturnTypeComponents(layerOp.input(), layerOp.mem_perm(), layerOp.dst_order(), inferredReturnShapes,
                                     false);

    VPUX_THROW_WHEN(inferredReturnShapes.size() != 1, "Should should be 1 but got {0}", inferredReturnShapes.size());

    auto dims = inferredReturnShapes.front().getDims();

    return mlir::SmallVector<int64_t>{dims.begin(), dims.end()};
}

//
// UniquifyBranches
//

class UniquifyBranches final : public IE::UniquifyBranchesBase<UniquifyBranches> {
public:
    explicit UniquifyBranches(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void UniquifyBranches::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MoveExpandBeforeSlice>(&ctx, _log);
    patterns.add<MoveReorderBeforeSlice>(&ctx, _log);
    patterns.add<MoveTransposeBeforeSlice>(&ctx, _log);
    patterns.add<MovePermuteCastBeforeSlice>(&ctx, _log);
    patterns.add<MoveAffineReshapeBeforeSlice>(&ctx, _log);
    patterns.add<MoveMemPermuteBeforeSlice>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createUniquifyBranchesPass(Logger log) {
    return std::make_unique<UniquifyBranches>(log);
}
