//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/enums.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

// ======================================================================================
// ConvertToReduceSumRewriter
class ConvertToReduceSumRewriter final : public mlir::OpRewritePattern<IE::ExtractImagePatchesOp> {
public:
    ConvertToReduceSumRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ExtractImagePatchesOp>(ctx, benefitLow), _log(log) {
        setDebugName("ConvertToReduceSumRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExtractImagePatchesOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

Const::DeclareOp createAxesTensor(mlir::Location loc, const mlir::SmallVector<int64_t>& axes,
                                  mlir::PatternRewriter& rewriter) {
    mlir::SmallVector<int32_t> axesI32 = to_small_vector(axes | transformed([](int64_t axis) {
                                                             return checked_cast<int32_t>(axis);
                                                         }));

    const auto tensorType =
            mlir::RankedTensorType::get({checked_cast<int32_t>(axesI32.size())},
                                        mlir::IntegerType::get(rewriter.getContext(), 32, mlir::IntegerType::Signed));
    const auto tensorAttr = mlir::DenseElementsAttr::get(tensorType, makeArrayRef(axesI32));
    const auto tensorContentAttr = Const::ContentAttr::get(tensorAttr);
    return rewriter.create<Const::DeclareOp>(loc, tensorType, tensorContentAttr);
}

bool isExtractImagePatchesJustTranspose(IE::ExtractImagePatchesOp op, Logger log) {
    if (op.sizes() == nullptr || op.strides() == nullptr || op.rates() == nullptr || op.autoPadAttr() == nullptr) {
        return false;
    }

    const auto dataShape = getShape(op.data());
    const auto sizes = parseIntArrayAttr<int64_t>(op.sizes());
    const auto strides = parseIntArrayAttr<int64_t>(op.strides());
    const auto rates = parseIntArrayAttr<int64_t>(op.rates());

    // Check that the ExtractImagePatrches does only a transpose
    if (sizes.size() != 2 || sizes[vpux::Dims4D::Kernel::Y.ind()] != 1 ||
        sizes[vpux::Dims4D::Kernel::X.ind()] != dataShape[vpux::Dims4D::Act::W]) {
        return false;
    }
    if (strides.size() != 2 ||
        (strides[vpux::Dims4D::Kernel::Y.ind()] != 1 || strides[vpux::Dims4D::Kernel::X.ind()] != 1)) {
        return false;
    }
    if (rates.size() != 2 || (rates[vpux::Dims4D::Kernel::Y.ind()] != 1 || rates[vpux::Dims4D::Kernel::X.ind()] != 1)) {
        return false;
    }
    if (op.autoPad() != IE::PadType::VALID) {
        return false;
    }
    if (dataShape[vpux::Dims4D::Act::C] != 1) {
        return false;
    }
    log.trace("ExtractImagePatches op is equivalent with a transpose op. - '{0}'", op->getLoc());
    return true;
}

mlir::LogicalResult replaceExtractImagePatchesWithTranspose(IE::ExtractImagePatchesOp& op,
                                                            mlir::PatternRewriter& rewriter, Logger log) {
    mlir::MLIRContext* ctx = op->getContext();
    // The equivalent permutation order
    log.trace("ExtractImagePatches op is replaced with a Transpose op - '{0}'", op->getLoc());
    const auto orderOutputAttr = mlir::AffineMapAttr::get(vpux::DimsOrder::NWHC.toAffineMap(ctx));
    rewriter.replaceOpWithNewOp<IE::TransposeOp>(op, op.getType(), op.data(), nullptr, orderOutputAttr);
    return mlir::success();
}

mlir::LogicalResult ConvertToReduceSumRewriter::matchAndRewrite(IE::ExtractImagePatchesOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // Check if ExtractImagePatches is equivalent with a single Transpose
    if (!isExtractImagePatchesJustTranspose(origOp, _log)) {
        return mlir::failure();
    }
    // Check if one of the following patterns appears:
    // 1. ReduceSum -> ExtractImagePatches -> Tranpose -> ReduceSum
    // 2. ReduceSum -> ExtractImagePatches -> ReduceSum
    auto aboveReduceSumOp = origOp.data().getDefiningOp<IE::ReduceSumOp>();
    if (aboveReduceSumOp != nullptr) {
        // The ReduceSum op should have only one consumer
        if (!aboveReduceSumOp.getResult().hasOneUse()) {
            return replaceExtractImagePatchesWithTranspose(origOp, rewriter, _log);
        }
        // The ExtractImagePatches op should have only one consumer
        if (!origOp.getResult().hasOneUse()) {
            return replaceExtractImagePatchesWithTranspose(origOp, rewriter, _log);
        }

        auto transposeOp = mlir::dyn_cast<IE::TransposeOp>(*(origOp.output().getUsers().begin()));
        vpux::IE::ReduceSumOp belowReduceSumOp;
        if (transposeOp != nullptr) {
            // The Transpose op should have only one consumer
            if (!transposeOp.getResult().hasOneUse()) {
                return replaceExtractImagePatchesWithTranspose(origOp, rewriter, _log);
            }
            belowReduceSumOp = mlir::dyn_cast<IE::ReduceSumOp>(*(transposeOp.output().getUsers().begin()));
        } else {
            belowReduceSumOp = mlir::dyn_cast<IE::ReduceSumOp>(*(origOp.output().getUsers().begin()));
        }

        if (belowReduceSumOp == nullptr) {
            return replaceExtractImagePatchesWithTranspose(origOp, rewriter, _log);
        }
        _log.trace("Found ReduceSum -> ExtractImagePatches -> [Tranpose] -> ReduceSum subgraph. - '{0}",
                   origOp->getLoc());

        // Additional checks for aboveReduceSumOp
        if (!aboveReduceSumOp.keep_dims() || aboveReduceSumOp.axes() == nullptr) {
            return replaceExtractImagePatchesWithTranspose(origOp, rewriter, _log);
        }

        auto aboveReduceSumOpAxes = parseIntArrayAttr<int64_t>(vpux::IE::getIntArrayAttrValue(aboveReduceSumOp.axes()));
        if (aboveReduceSumOpAxes.size() > 1 || aboveReduceSumOpAxes[0] != vpux::Dims4D::Act::C.ind()) {
            return replaceExtractImagePatchesWithTranspose(origOp, rewriter, _log);
        }

        // Additional checks for belowReduceSumOp
        auto belowReduceSumOpAxes = parseIntArrayAttr<int64_t>(vpux::IE::getIntArrayAttrValue(belowReduceSumOp.axes()));
        if (belowReduceSumOp.axes() == nullptr) {
            return replaceExtractImagePatchesWithTranspose(origOp, rewriter, _log);
        }
        if (transposeOp == nullptr) {
            if (belowReduceSumOpAxes.size() > 1 || belowReduceSumOpAxes[0] != vpux::Dims4D::Act::C.ind()) {
                return replaceExtractImagePatchesWithTranspose(origOp, rewriter, _log);
            }
        } else {
            const auto transposeOpPerm = vpux::DimsOrder::fromAffineMap(transposeOp.order_value().getValue());
            if (transposeOpPerm != vpux::DimsOrder::NHWC || belowReduceSumOpAxes.size() > 1 ||
                belowReduceSumOpAxes[0] != vpux::Dims4D::Act::W.ind()) {
                return replaceExtractImagePatchesWithTranspose(origOp, rewriter, _log);
            }
        }
        _log.trace("Confirmed that the subgraph satisfies the rest of constrains. - '{0}", origOp->getLoc());

        // Replace the subgraph with the equivalent ReduceSum op
        aboveReduceSumOpAxes.push_back(vpux::Dims4D::Act::W.ind());
        // Create the tensor containing the fused reduction axes
        auto axesOp = createAxesTensor(origOp->getLoc(), aboveReduceSumOpAxes, rewriter);
        auto newReduceSumOp =
                rewriter.create<IE::ReduceSumOp>(origOp->getLoc(), aboveReduceSumOp.input(), axesOp.output(), false);
        mlir::SmallVector<int64_t> unsqueezeAxis{vpux::Dims4D::Act::H.ind()};
        if (belowReduceSumOp.keep_dims()) {
            unsqueezeAxis.push_back(belowReduceSumOpAxes[0]);
        }
        const auto unsqueezeAxesAttr = getIntArrayAttr(getContext(), unsqueezeAxis);
        rewriter.replaceOpWithNewOp<IE::UnsqueezeOp>(belowReduceSumOp, newReduceSumOp, nullptr, unsqueezeAxesAttr);
        return mlir::success();

    } else {
        return replaceExtractImagePatchesWithTranspose(origOp, rewriter, _log);
    }
    return mlir::failure();
}

// ======================================================================================
// ConvertToSliceConcatRewriter
class ConvertToSliceConcatRewriter final : public mlir::OpRewritePattern<IE::ExtractImagePatchesOp> {
public:
    ConvertToSliceConcatRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ExtractImagePatchesOp>(ctx, benefitLow), _log(log) {
        setDebugName("ConvertToSliceConcatRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExtractImagePatchesOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertToSliceConcatRewriter::matchAndRewrite(IE::ExtractImagePatchesOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // Check if ExtractImagePatches is equivalent with with NxSlice->Concat sequence
    if (origOp.sizes() == nullptr || origOp.strides() == nullptr || origOp.rates() == nullptr ||
        origOp.autoPadAttr() == nullptr) {
        return mlir::failure();
    }

    const auto dataShape = getShape(origOp.data());
    const auto sizes = parseIntArrayAttr<int64_t>(origOp.sizes());
    const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());
    const auto rates = parseIntArrayAttr<int64_t>(origOp.rates());

    // Check that the ExtractImagePatrches does only a transpose
    if (sizes.size() != 2 || sizes[vpux::Dims4D::Kernel::Y.ind()] == 1 ||
        sizes[vpux::Dims4D::Kernel::Y.ind()] >= dataShape[vpux::Dims4D::Act::H] ||
        sizes[vpux::Dims4D::Kernel::X.ind()] != dataShape[vpux::Dims4D::Act::W]) {
        return mlir::failure();
    }
    if (strides.size() != 2 ||
        (strides[vpux::Dims4D::Kernel::Y.ind()] != 1 || strides[vpux::Dims4D::Kernel::X.ind()] != 1)) {
        return mlir::failure();
    }
    if (rates.size() != 2 || (rates[vpux::Dims4D::Kernel::Y.ind()] != 1 || rates[vpux::Dims4D::Kernel::X.ind()] != 1)) {
        return mlir::failure();
    }
    if (origOp.autoPad() != IE::PadType::VALID) {
        return mlir::failure();
    }
    if (dataShape[vpux::Dims4D::Act::C] != 1) {
        return mlir::failure();
    }
    _log.trace("ExtractImagePatches op is equivalent with nxSliceOp -> Concat. - '{0}'", origOp->getLoc());

    // Check if one of the following pattern is met:
    // 1. ExtractImagePatches -> Transpose-> AffineReshape
    // 2. ExtractImagePatches -> Transpose
    // If ExtractImagePatches -> Transpose-> AffineReshape pattern + all required constrains are met and:
    // If pattern 1 is met, do the rewrite:
    // ExtractImagePatches -> Transpose-> AffineReshape  =>  NxSlice -> Concat
    // If pattern 2 is met, do the rewrite:
    // ExtractImagePatches -> Transpose                  =>  NxSlice -> Concat -> AffineReshape
    if (!origOp.getResult().hasOneUse()) {
        return mlir::failure();
    }

    auto transposeOp = mlir::dyn_cast<IE::TransposeOp>(*(origOp.output().getUsers().begin()));
    if (transposeOp == nullptr) {
        return mlir::failure();
    }

    const auto origPerm = vpux::DimsOrder::fromAffineMap(transposeOp.order_value().getValue());
    if (origPerm != vpux::DimsOrder::NHWC) {
        return mlir::failure();
    }
    _log.trace("ExtractImagePatches -> Transpose pattern with all required constrains satisfied met. - '{0}'",
               origOp.getLoc());

    const auto dataHeight = dataShape[vpux::Dims4D::Act::H];
    const auto sizesHeight = (parseIntArrayAttr<int64_t>(origOp.sizes()))[vpux::Dims4D::Kernel::Y.ind()];
    const auto numberOfSlices = checked_cast<size_t>(dataHeight - sizesHeight + 1);
    mlir::SmallVector<mlir::Value> sliceOpValues;

    if (transposeOp.output().getUsers().empty()) {
        return mlir::failure();
    }
    auto affineReshapeOp = mlir::dyn_cast<IE::AffineReshapeOp>(*(transposeOp.output().getUsers().begin()));

    // Backup function to do the corresponding rewrite of pattern 2
    auto rewriteExtractImagePatchesTranspose = [&](bool suitableReshape) -> mlir::LogicalResult {
        mlir::MLIRContext* ctx = origOp->getContext();
        for (size_t idx = 0; idx < numberOfSlices; idx++) {
            auto staticOffsets = mlir::SmallVector<int64_t>(dataShape.size(), 0);
            staticOffsets[vpux::Dims4D::Act::H.ind()] = idx;
            mlir::SmallVector<int64_t> staticSizes = to_small_vector(dataShape.raw());
            staticSizes[vpux::Dims4D::Act::H.ind()] = sizesHeight;
            auto sliceOp =
                    rewriter.create<IE::SliceOp>(origOp->getLoc(), origOp.data(), getIntArrayAttr(ctx, staticOffsets),
                                                 getIntArrayAttr(ctx, staticSizes));
            sliceOpValues.push_back(sliceOp.result());
        }
        if (suitableReshape) {
            rewriter.replaceOpWithNewOp<IE::ConcatOp>(affineReshapeOp, sliceOpValues, vpux::Dims4D::Act::C);
        } else {
            auto concatOp = rewriter.create<IE::ConcatOp>(origOp->getLoc(), sliceOpValues, vpux::Dims4D::Act::C);

            const auto transposeShape = transposeOp.getType().getShape();
            const auto transposeShapeAttr = getIntArrayAttr(ctx, transposeShape);

            rewriter.replaceOpWithNewOp<IE::ReshapeOp>(transposeOp, concatOp, nullptr, false, transposeShapeAttr);
        }
        return mlir::success();
    };

    auto isSupportedReshape = [&]() -> bool {
        // Check if pattern 1 is met and do the rewrite
        if (affineReshapeOp != nullptr) {
            // Check that the Transpose op just 1 child
            if (!transposeOp.getResult().hasOneUse()) {
                return false;
            }
            const auto affineReshapeDimMapping = parseIntArrayOfArrayAttr<int64_t>(affineReshapeOp.dim_mappingAttr());
            const auto affineReshapeShapeValue = parseIntArrayAttr<int64_t>(affineReshapeOp.shape_valueAttr());
            // AffineReshape conditions
            if (affineReshapeDimMapping.size() != 4) {
                return false;
            }
            if (affineReshapeDimMapping[vpux::Dims4D::Act::N.ind()].size() != 1 ||
                affineReshapeDimMapping[vpux::Dims4D::Act::N.ind()][0] != 0) {
                return false;
            }
            if (affineReshapeDimMapping[vpux::Dims4D::Act::C.ind()].size() != 1 ||
                affineReshapeDimMapping[vpux::Dims4D::Act::C.ind()][0] != 1) {
                return false;
            }
            if (affineReshapeDimMapping[vpux::Dims4D::Act::H.ind()].size() != 1 ||
                affineReshapeDimMapping[vpux::Dims4D::Act::H.ind()][0] != 1) {
                return false;
            }
            if (affineReshapeDimMapping[vpux::Dims4D::Act::W.ind()].size() != 2 ||
                affineReshapeDimMapping[vpux::Dims4D::Act::W.ind()][0] != 2 ||
                affineReshapeDimMapping[vpux::Dims4D::Act::W.ind()][1] != 3) {
                return false;
            }

            if (affineReshapeShapeValue.size() != 4) {
                return false;
            }
            if (affineReshapeShapeValue[vpux::Dims4D::Act::N.ind()] != 1) {
                return false;
            }
            if (affineReshapeShapeValue[vpux::Dims4D::Act::C.ind()] != checked_cast<int64_t>(numberOfSlices)) {
                return false;
            }
            if (affineReshapeShapeValue[vpux::Dims4D::Act::H.ind()] != sizes[vpux::Dims4D::Kernel::Y.ind()]) {
                return false;
            }
            if (affineReshapeShapeValue[vpux::Dims4D::Act::W.ind()] != sizes[vpux::Dims4D::Kernel::X.ind()]) {
                return false;
            }
            _log.trace("All conditions of ExtractImagePatches -> Transpose-> AffineReshape pattern were met. - '{0}'",
                       origOp->getLoc());
            return true;
        }
        return false;
    };
    return rewriteExtractImagePatchesTranspose(isSupportedReshape());
}

//
// ConvertExtractImagePatchesPass
//

class ConvertExtractImagePatchesPass final : public IE::ConvertExtractImagePatchesBase<ConvertExtractImagePatchesPass> {
public:
    explicit ConvertExtractImagePatchesPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void ConvertExtractImagePatchesPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertToReduceSumRewriter>(&ctx, _log);
    patterns.add<ConvertToSliceConcatRewriter>(&ctx, _log);
    IE::ReshapeOp::getCanonicalizationPatterns(patterns, &ctx);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertExtractImagePatchesPass
//
std::unique_ptr<mlir::Pass> vpux::IE::createConvertExtractImagePatchesPass(Logger log) {
    return std::make_unique<ConvertExtractImagePatchesPass>(log);
}
