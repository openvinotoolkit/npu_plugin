//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;
mlir::LogicalResult vpux::IE::MVN6Op::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::MVN6OpAdaptor mvn(operands, attrs);
    if (mlir::failed(mvn.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mvn.input().getType().cast<mlir::ShapedType>();
    const auto rankedInType = mvn.input().getType().cast<mlir::RankedTensorType>();
    const auto outDesc = IE::getTensorAttr(rankedInType);
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType(), outDesc);

    return mlir::success();
}

namespace {

//
// getAxes
//

mlir::FailureOr<SmallVector<int64_t>> getAxes(IE::MVN6OpAdaptor mvn, mlir::Location loc) {
    if (mvn.axes() != nullptr && mvn.axes_value().hasValue()) {
        return errorAt(loc, "Ambiguous axes representation");
    }
    if (mvn.axes() == nullptr && !mvn.axes_value().hasValue()) {
        return errorAt(loc, "Missing axes value");
    }

    if (mvn.axes_value().hasValue()) {
        return parseIntArrayAttr<int64_t>(mvn.axes_value().getValue());
    }

    auto axesConst = mvn.axes().getDefiningOp<Const::DeclareOp>();
    if (axesConst == nullptr) {
        return errorAt(loc, "Only constant axes are supported");
    }

    const auto axesContent = axesConst.content();
    auto axes = to_small_vector(axesContent.getValues<int64_t>());

    const auto inType = mvn.input().getType().cast<mlir::ShapedType>();
    const auto inRank = inType.getRank();

    for (auto& axis : axes) {
        if (axis < 0) {
            axis += inRank;
        }
    }
    std::sort(axes.begin(), axes.end());

    return axes;
}

//
// ConvertConstToAttr
//

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::MVN6Op> {
public:
    using mlir::OpRewritePattern<IE::MVN6Op>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::MVN6Op origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::MVN6Op origOp, mlir::PatternRewriter& rewriter) const {
    if (origOp.axes_value().hasValue()) {
        return mlir::failure();
    }

    const auto axes = getAxes(origOp, origOp->getLoc());
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    const auto axesAttr = getIntArrayAttr(getContext(), axes.getValue());
    rewriter.replaceOpWithNewOp<IE::MVN6Op>(origOp, origOp.input(), nullptr, axesAttr, origOp.normalize_varianceAttr(),
                                            origOp.epsAttr(), origOp.eps_modeAttr());
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::MVN6Op::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
