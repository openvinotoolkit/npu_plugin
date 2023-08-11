//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::InterpolateOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::InterpolateOpAdaptor interpolate(operands, attrs);
    if (mlir::failed(interpolate.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = interpolate.input().getType().cast<mlir::ShapedType>();

    auto outShape = IE::calcOutputShapes(loc, interpolate, Logger::global());

    if (mlir::failed(outShape)) {
        return mlir::failure();
    }

    inferredReturnShapes.emplace_back(outShape.getValue(), inType.getElementType());
    return mlir::success();
}

namespace {

//
// ConvertInputsToAttr
//

class ConvertInputsToAttr final : public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    using mlir::OpRewritePattern<IE::InterpolateOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp InterpolateOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertInputsToAttr::matchAndRewrite(IE::InterpolateOp InterpolateOp,
                                                         mlir::PatternRewriter& rewriter) const {
    if (InterpolateOp.sizes_attr().hasValue() || InterpolateOp.scales_attr().hasValue() ||
        InterpolateOp.axes_attr().hasValue()) {
        return mlir::failure();
    }

    // convert sizes
    auto sizes = IE::extractIntVector(InterpolateOp.getLoc(), InterpolateOp.sizes(), None);

    if (mlir::failed(sizes)) {
        return mlir::failure();
    }
    const auto sizesAttr = getIntArrayAttr(InterpolateOp.getContext(), sizes.getValue());

    // convert scales
    auto scales = IE::extractFPVector(InterpolateOp.getLoc(), InterpolateOp.scales(), None);

    if (mlir::failed(scales)) {
        return mlir::failure();
    }
    const auto scalesAttr = getFPArrayAttr(InterpolateOp.getContext(), scales.getValue());

    // convert axes
    auto axes = (InterpolateOp.axes() == nullptr)
                        ? IE::getDefaultInterpolateAxes(InterpolateOp)
                        : IE::extractIntVector(InterpolateOp.getLoc(), InterpolateOp.axes(), None);

    if (mlir::failed(axes)) {
        return mlir::failure();
    }
    const auto axesAttr = getIntArrayAttr(InterpolateOp.getContext(), axes.getValue());

    // rewrite layer
    rewriter.replaceOpWithNewOp<IE::InterpolateOp>(
            InterpolateOp, InterpolateOp.input(), nullptr, nullptr, nullptr, sizesAttr, scalesAttr, axesAttr,
            InterpolateOp.tile_offset_attrAttr(), InterpolateOp.initial_input_dims_attrAttr(),
            InterpolateOp.initial_output_dims_attrAttr(), InterpolateOp.attr());

    return mlir::success();
}

class ConvertInputToFP16 final : public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    using mlir::OpRewritePattern<IE::InterpolateOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp Op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertInputToFP16::matchAndRewrite(IE::InterpolateOp op, mlir::PatternRewriter& rewriter) const {
    const auto inputType = op.input().getType().cast<mlir::ShapedType>().getElementType();

    if (inputType.isUnsignedInteger(8)) {
        auto convertOpBefore =
                rewriter.create<IE::ConvertOp>(op.getLoc(), op.input(), mlir::Float16Type::get(getContext()));
        auto interpolateOp = rewriter.create<IE::InterpolateOp>(
                op.getLoc(), convertOpBefore.output(), op.sizes(), op.scales(), op.axes(), op.sizes_attrAttr(),
                op.scales_attrAttr(), op.axes_attrAttr(), op.tile_offset_attrAttr(), op.initial_input_dims_attrAttr(),
                op.initial_output_dims_attrAttr(), op.attr());

        rewriter.replaceOpWithNewOp<IE::ConvertOp>(op, interpolateOp.output(), inputType);
        return mlir::success();
    }

    return mlir::failure();
}

class ConvertToNearest final : public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    using mlir::OpRewritePattern<IE::InterpolateOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp Op, mlir::PatternRewriter& rewriter) const final;
};

// Convert any type of Interpolate that is like BroadCast axes to NEAREST Interpolate and with ASYMMETRIC CoordMode
// For example: inShape: 1x16x1x1, outShape: 1x16x32x32, broadcast at H and W dimensions
// This kind of NEAREST Interpolate will further optimization at `ConvertNearestToBroadcastOrStridedConcatPass`
//  - Convert to NCEInterpolateOp that can be executed using the Storage Element hardware feature
//  - Convert to BroadCastOp and further convert to TileOp, finally to PerAxisTileDMAOp
mlir::LogicalResult ConvertToNearest::matchAndRewrite(IE::InterpolateOp op, mlir::PatternRewriter& rewriter) const {
    auto* ctx = op->getContext();
    if (!IE::isBroadCastInterpolate(op)) {
        return mlir::failure();
    }

    const auto originalAttr = op.attr();
    if (originalAttr.getMode().getValue() == IE::InterpolateMode::NEAREST &&
        originalAttr.getCoordMode().getValue() == IE::InterpolateCoordMode::ASYMMETRIC) {
        return mlir::failure();
    }

    const auto newInterpolateAttr = IE::InterpolateAttr::get(
            ctx, IE::InterpolateModeAttr::get(ctx, IE::InterpolateMode::NEAREST), originalAttr.getShapeCalcMode(),
            IE::InterpolateCoordModeAttr::get(ctx, IE::InterpolateCoordMode::ASYMMETRIC), originalAttr.getNearestMode(),
            originalAttr.getAntialias(), originalAttr.getPadsBegin(), originalAttr.getPadsEnd(),
            originalAttr.getCubeCoeff());

    rewriter.replaceOpWithNewOp<IE::InterpolateOp>(op, op.input(), op.sizes(), op.scales(), op.axes(),
                                                   op.sizes_attrAttr(), op.scales_attrAttr(), op.axes_attrAttr(),
                                                   op.tile_offset_attrAttr(), op.initial_input_dims_attrAttr(),
                                                   op.initial_output_dims_attrAttr(), newInterpolateAttr);

    return mlir::success();
}

}  // namespace

//
// fold
//

mlir::OpFoldResult vpux::IE::InterpolateOp::fold(ArrayRef<mlir::Attribute>) {
    if (input().getType() == output().getType()) {
        return input();
    }

    return nullptr;
}

//
// getCanonicalizationPatterns
//

void vpux::IE::InterpolateOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                          mlir::MLIRContext* context) {
    patterns.add<ConvertInputsToAttr>(context);
    patterns.add<ConvertInputToFP16>(context);
    patterns.add<ConvertToNearest>(context);
}
