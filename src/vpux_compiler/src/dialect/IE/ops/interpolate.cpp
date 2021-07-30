//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

namespace {

mlir::FailureOr<SmallVector<int64_t>> extractIntVector(mlir::Location loc, const mlir::Value& value,
                                                       const mlir::ArrayAttr& attr) {
    if (attr != nullptr) {
        return parseIntArrayAttr<int64_t>(attr);
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
                                                     const mlir::ArrayAttr& attr) {
    if (attr != nullptr) {
        return parseFPArrayAttr<double>(attr);
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

mlir::FailureOr<SmallVector<int64_t>> calcOutputShapes(mlir::Location loc, IE::InterpolateOpAdaptor interpolate) {
    const auto axes = extractIntVector(loc, interpolate.axes(), interpolate.axes_attr());
    const auto axes_val = axes.getValue();

    const auto inType = interpolate.input().getType().cast<mlir::ShapedType>();
    const auto inputShape = inType.getShape();

    SmallVector<int64_t> outShape;

    for (size_t i = 0; i < inputShape.size(); ++i) {
        outShape.emplace_back(inputShape[i]);
    }

    auto calcMode = interpolate.attr().shape_calc_mode().getValue();

    if (calcMode == IE::InterpolateCalcMode::sizes) {
        const auto sizes = extractIntVector(loc, interpolate.sizes(), interpolate.sizes_attr());
        const auto sizes_val = sizes.getValue();

        if (sizes_val.size() != axes_val.size()) {
            return errorAt(loc,
                           "Num of elements in sizes tensor: {0} should be equal to number of indices in axes: {1}",
                           sizes_val.size(), axes_val.size());
        }
        auto sizesIter = sizes_val.begin();

        for (const auto& i : axes_val)
            outShape[i] = *sizesIter++;

        return outShape;

    } else if (calcMode == IE::InterpolateCalcMode::scales) {
        const auto scales = extractFPVector(loc, interpolate.scales(), interpolate.scales_attr());
        const auto scales_val = scales.getValue();

        if (scales_val.size() != axes_val.size()) {
            return errorAt(loc,
                           "Num of elements in scales tensor: {0} should be equal to number of indices in axes: {1}",
                           scales_val.size(), axes_val.size());
        }

        auto scalesIter = scales_val.begin();

        for (const auto& i : axes_val) {
            outShape[i] = static_cast<int64_t>(floor((*scalesIter++) * inputShape[i]));
        }

        return outShape;

    } else
        return errorAt(loc, "Doesn't support shape_calculation_mode: {0}", calcMode);
}

}  // namespace

mlir::LogicalResult vpux::IE::InterpolateOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::InterpolateOpAdaptor interpolate(operands, attrs);
    if (mlir::failed(interpolate.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = interpolate.input().getType().cast<mlir::ShapedType>();

    auto outShape = calcOutputShapes(loc, interpolate);

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
    auto sizes = extractIntVector(InterpolateOp.getLoc(), InterpolateOp.sizes(), nullptr);

    if (mlir::failed(sizes)) {
        return mlir::failure();
    }
    const auto sizesAttr = getIntArrayAttr(InterpolateOp.getContext(), sizes.getValue());

    // convert scales
    auto scales = extractFPVector(InterpolateOp.getLoc(), InterpolateOp.scales(), nullptr);

    if (mlir::failed(scales)) {
        return mlir::failure();
    }
    const auto scalesAttr = getFPArrayAttr(InterpolateOp.getContext(), scales.getValue());

    // convert axes
    auto axes = extractIntVector(InterpolateOp.getLoc(), InterpolateOp.axes(), nullptr);

    if (mlir::failed(axes)) {
        return mlir::failure();
    }
    const auto axesAttr = getIntArrayAttr(InterpolateOp.getContext(), axes.getValue());

    // rewrite layer
    rewriter.replaceOpWithNewOp<IE::InterpolateOp>(InterpolateOp, InterpolateOp.input(), nullptr, nullptr, nullptr,
                                                   sizesAttr, scalesAttr, axesAttr, InterpolateOp.attr());

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::InterpolateOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns,
                                                          mlir::MLIRContext* context) {
    patterns.insert<ConvertInputsToAttr>(context);
}
