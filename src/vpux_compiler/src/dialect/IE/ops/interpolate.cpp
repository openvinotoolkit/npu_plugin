//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include <numeric>

using namespace vpux;

namespace {

mlir::FailureOr<SmallVector<int64_t>> extractIntVector(mlir::Location loc, const mlir::Value& value,
                                                       const Optional<mlir::ArrayAttr>& attr) {
    if (attr.hasValue() && attr.getValue() != nullptr) {
        return parseIntArrayAttr<int64_t>(attr.getValue());
    } else if (value != nullptr) {
        auto valueConst = value.getDefiningOp<Const::DeclareOp>();
        if (valueConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for interpolate attribute");
        }

        const auto valueContent = valueConst.content();
        return to_small_vector(valueContent.getValues<int64_t>());
    }
    return errorAt(loc, "Parameter were not provided");
}

mlir::FailureOr<SmallVector<double>> extractFPVector(mlir::Location loc, const mlir::Value& value,
                                                     const Optional<mlir::ArrayAttr>& attr) {
    if (attr.hasValue() && attr.getValue() != nullptr) {
        return parseFPArrayAttr<double>(attr.getValue());
    } else if (value != nullptr) {
        auto valueConst = value.getDefiningOp<Const::DeclareOp>();

        if (valueConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for interpolate attribute");
        }

        const auto valueContent = valueConst.content();
        return to_small_vector(valueContent.getValues<double>());
    }
    return errorAt(loc, "Parameter were not provided");
}

void applyInterpPads(MutableArrayRef<int64_t> outShape, ArrayRef<int64_t> padsBegin, ArrayRef<int64_t> padsEnd) {
    // pads might be zero initialized
    if (padsBegin.size() != padsEnd.size() || padsBegin.size() != outShape.size()) {
        return;
    }
    // naive implementation only apply pads to calculated output shape
    for (auto d : outShape | indexed) {
        d.value() += padsBegin[d.index()] + padsEnd[d.index()];
    }
}

mlir::FailureOr<SmallVector<int64_t>> propagateShape(mlir::Location loc, mlir::FailureOr<SmallVector<int64_t>> axes,
                                                     ArrayRef<int64_t> origShape,
                                                     mlir::FailureOr<ArrayRef<int64_t>> padsBegin,
                                                     mlir::FailureOr<ArrayRef<int64_t>> padsEnd,
                                                     vpux::IE::InterpolateCalcMode calcMode,
                                                     mlir::FailureOr<ArrayRef<int64_t>> sizes,
                                                     mlir::FailureOr<ArrayRef<double>> scales, vpux::Logger log) {
    log.trace("Interp propagate shape: input = {0}", origShape);
    const auto axes_val = axes.getValue();
    auto inferedShape = to_small_vector(origShape);

    if (calcMode == IE::InterpolateCalcMode::SIZES) {
        const auto sizes_val = sizes.getValue();

        if (sizes_val.size() != axes_val.size()) {
            return errorAt(loc,
                           "Num of elements in sizes tensor: {0} should be equal to number of indices in axes: {1}",
                           sizes_val.size(), axes_val.size());
        }
        auto sizesIter = sizes_val.begin();

        for (const auto& i : axes_val) {
            log.trace("Interp sizes - axis: {0}", i);
            inferedShape[i] = *sizesIter++;
        }
    } else if (calcMode == IE::InterpolateCalcMode::SCALES) {
        const auto scales_val = scales.getValue();

        if (scales_val.size() != axes_val.size()) {
            return errorAt(loc,
                           "Num of elements in scales tensor: {0} should be equal to number of indices in axes: {1}",
                           scales_val.size(), axes_val.size());
        }

        auto scalesIter = scales_val.begin();

        for (const auto& i : axes_val) {
            log.trace("Interp scales - axis: {0}", i);
            inferedShape[i] = static_cast<int64_t>(floor((*scalesIter++) * origShape[i]));
        }

    } else
        return errorAt(loc, "Doesn't support shape_calculation_mode: {0}", calcMode);

    // meaning pads provided in attributes
    if (mlir::succeeded(padsBegin) && mlir::succeeded(padsEnd)) {
        applyInterpPads(inferedShape, padsBegin.getValue(), padsEnd.getValue());
    }

    log.trace("Interp propagate shape: output = {0}", inferedShape);

    return inferedShape;
}

SmallVector<int64_t> getDefaultInterpolateAxes(IE::InterpolateOpAdaptor interpolate) {
    SmallVector<int64_t> axes(interpolate.input().getType().cast<mlir::ShapedType>().getRank());
    std::iota(axes.begin(), axes.end(), 0);

    return axes;
}

mlir::FailureOr<SmallVector<int64_t>> calcOutputShapes(mlir::Location loc, IE::InterpolateOpAdaptor interpolate,
                                                       vpux::Logger log) {
    const auto axesAttr = interpolate.axes_attr();
    const bool validAxesAttr = (axesAttr.hasValue() && axesAttr.getValue() != nullptr);
    const auto axes = (interpolate.axes() == nullptr && !validAxesAttr)
                              ? getDefaultInterpolateAxes(interpolate)
                              : extractIntVector(loc, interpolate.axes(), interpolate.axes_attr());
    const auto beginPads = extractIntVector(loc, {}, interpolate.attr().pads_begin());
    const auto endPads = extractIntVector(loc, {}, interpolate.attr().pads_end());

    const auto inType = interpolate.input().getType().cast<mlir::ShapedType>();
    const auto inputShape = inType.getShape();

    return propagateShape(loc, axes, inputShape, beginPads, endPads, interpolate.attr().shape_calc_mode().getValue(),
                          extractIntVector(loc, interpolate.sizes(), interpolate.sizes_attr()),
                          extractFPVector(loc, interpolate.scales(), interpolate.scales_attr()), log);
}

}  // namespace

mlir::LogicalResult vpux::IE::InterpolateOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::InterpolateOpAdaptor interpolate(operands, attrs);
    if (mlir::failed(interpolate.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = interpolate.input().getType().cast<mlir::ShapedType>();

    auto outShape = calcOutputShapes(loc, interpolate, Logger::global());

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
    auto sizes = extractIntVector(InterpolateOp.getLoc(), InterpolateOp.sizes(), None);

    if (mlir::failed(sizes)) {
        return mlir::failure();
    }
    const auto sizesAttr = getIntArrayAttr(InterpolateOp.getContext(), sizes.getValue());

    // convert scales
    auto scales = extractFPVector(InterpolateOp.getLoc(), InterpolateOp.scales(), None);

    if (mlir::failed(scales)) {
        return mlir::failure();
    }
    const auto scalesAttr = getFPArrayAttr(InterpolateOp.getContext(), scales.getValue());

    // convert axes
    auto axes = (InterpolateOp.axes() == nullptr)
                        ? getDefaultInterpolateAxes(InterpolateOp)
                        : extractIntVector(InterpolateOp.getLoc(), InterpolateOp.axes(), None);

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
}
