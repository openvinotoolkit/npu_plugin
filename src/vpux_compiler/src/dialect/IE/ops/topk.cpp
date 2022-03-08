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

mlir::LogicalResult vpux::IE::TopKOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::TopKOpAdaptor topK(operands, attrs);
    if (mlir::failed(topK.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = topK.input().getType().cast<mlir::ShapedType>();
    const auto inputShape = inType.getShape();

    auto kConst = topK.k().getDefiningOp<Const::DeclareOp>();
    if (kConst == nullptr) {
        return errorAt(loc, "Only constant input is supported for k");
    }

    const auto kContent = kConst.content();
    if (!kContent.isSplat()) {
        return errorAt(loc, "K input must be scalar");
    }

    SmallVector<int64_t> outShape;
    for (size_t i = 0; i < inputShape.size(); ++i) {
        outShape.push_back(inputShape[i]);
    }
    int64_t axis = topK.axis().getInt();
    const auto inRank = inType.getRank();
    if (axis < 0) {
        axis += inRank;
    }
    outShape[axis] = kContent.getSplatValue<int64_t>();

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());
    inferredReturnShapes.emplace_back(outShape, topK.element_type().getValue());

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::TopKOp> {
public:
    using mlir::OpRewritePattern<IE::TopKOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::TopKOp topKOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::TopKOp topKOp, mlir::PatternRewriter& rewriter) const {
    // Translate k input to k attr.
    if (topKOp.k_value().hasValue()) {
        return mlir::failure();
    }

    auto kConst = topKOp.k().getDefiningOp<Const::DeclareOp>();
    const auto kContent = kConst.content();
    mlir::FailureOr<int64_t> k = kContent.getSplatValue<int64_t>();
    if (mlir::failed(k)) {
        return mlir::failure();
    }
    const auto kAttr = getIntAttr(topKOp.getContext(), k.getValue());

    rewriter.replaceOpWithNewOp<IE::TopKOp>(topKOp, topKOp.input(), topKOp.k(), kAttr, topKOp.axisAttr(),
                                            topKOp.modeAttr(), topKOp.sortAttr(), topKOp.element_typeAttr());

    return mlir::success();
}

}  // namespace

void vpux::IE::TopKOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.insert<ConvertConstToAttr>(context);
}
