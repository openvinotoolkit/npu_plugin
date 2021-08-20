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
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// getAxes
//

namespace {

mlir::FailureOr<SmallVector<int64_t>> getAxes(IE::ReduceMeanOpAdaptor reduceMean, mlir::Location loc) {
    if (reduceMean.axes() != nullptr && reduceMean.axes_value() != nullptr) {
        return errorAt(loc, "Ambiguous axes representation");
    }
    if (reduceMean.axes() == nullptr && reduceMean.axes_value() == nullptr) {
        return errorAt(loc, "Missed axes representation");
    }

    if (reduceMean.axes_value() != nullptr) {
        return parseIntArrayAttr<int64_t>(reduceMean.axes_value());
    }

    auto axesConst = reduceMean.axes().getDefiningOp<Const::DeclareOp>();
    if (axesConst == nullptr) {
        return errorAt(loc, "Only constant axes are supported");
    }

    const auto axesContent = axesConst.content();
    auto axes = to_small_vector(axesContent.getValues<int64_t>());
    std::sort(axes.begin(), axes.end());

    const auto inType = reduceMean.input().getType().cast<mlir::ShapedType>();
    const auto inRank = inType.getRank();

    for (auto& axis : axes) {
        if (axis < 0) {
            axis += inRank;
        }
    }

    return axes;
}

}  // namespace

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::ReduceMeanOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        mlir::SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ReduceMeanOpAdaptor reduceMean(operands, attrs);
    if (mlir::failed(reduceMean.verify(loc))) {
        return mlir::failure();
    }

    const auto axes = getAxes(reduceMean, loc);
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    // axes has only one element
    auto axis = axes->front();
    const auto inType = reduceMean.input().getType().cast<mlir::ShapedType>();
    const auto inShape = inType.getShape();

    const auto keepDims = reduceMean.keep_dims().getValue();

    SmallVector<int64_t> outShape;
    for (auto inInd : irange(inType.getRank())) {
        if (inInd == axis) {
            if (keepDims) {
                outShape.push_back(1);
            } else {
                continue;
            }
        }
        outShape.push_back(inShape[inInd]);
    }

    inferredReturnShapes.emplace_back(makeArrayRef(outShape), inType.getElementType());
    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::ReduceMeanOp> {
public:
    using mlir::OpRewritePattern<IE::ReduceMeanOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ReduceMeanOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::ReduceMeanOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    if (origOp.axes_value().hasValue()) {
        return mlir::failure();
    }

    const auto axes = getAxes(origOp, origOp->getLoc());
    if (mlir::failed(axes)) {
        return mlir::failure();
    }
    const auto axesAttr = getIntArrayAttr(getContext(), axes.getValue());

    rewriter.replaceOpWithNewOp<IE::ReduceMeanOp>(origOp, origOp.input(), nullptr, axesAttr, origOp.keep_dimsAttr());
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::ReduceMeanOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns,
                                                         mlir::MLIRContext* context) {
    patterns.insert<ConvertConstToAttr>(context);
}
