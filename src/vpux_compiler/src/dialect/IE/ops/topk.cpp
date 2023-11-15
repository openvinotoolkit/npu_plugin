//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes_utils.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

//
// verify
//

mlir::LogicalResult vpux::IE::TopKOp::verify() {
    if (!k()) {
        return mlir::success();
    }

    auto kNumElements = k().getType().cast<vpux::NDTypeInterface>().getNumElements();
    if (kNumElements != 1) {
        return errorAt(*this, "K should have only 1 element, while it has {0}", kNumElements);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::TopKOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::TopKOpAdaptor topK(operands, attrs);
    if (mlir::failed(topK.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = topK.input().getType().cast<mlir::ShapedType>();
    const auto inputShape = inType.getShape();

    const auto kValue = getConstOrAttrValue(topK.k(), topK.k_valueAttr());

    if (mlir::failed(kValue)) {
        return mlir::failure();
    }

    SmallVector<int64_t> outShape;
    for (size_t i = 0; i < inputShape.size(); ++i) {
        outShape.push_back(inputShape[i]);
    }
    int64_t axis = topK.axis();
    const auto inRank = inType.getRank();
    if (axis < 0) {
        axis += inRank;
    }

    outShape[axis] = kValue.value();

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());
    inferredReturnShapes.emplace_back(outShape, topK.element_type());

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
    auto arch = VPU::getArch(topKOp->getParentOfType<mlir::ModuleOp>());
    if (!(arch == VPU::ArchKind::VPUX37XX)) {
        return mlir::failure();
    }

    if (topKOp.k_value()) {
        return mlir::failure();
    }

    const auto kValue = getConstOrAttrValue(topKOp.k(), topKOp.k_valueAttr());

    if (mlir::failed(kValue)) {
        return mlir::failure();
    }

    const auto kValueAttr = getIntAttr(rewriter.getContext(), kValue.value());

    rewriter.replaceOpWithNewOp<IE::TopKOp>(topKOp, topKOp.input(), nullptr, kValueAttr, topKOp.axis(), topKOp.mode(),
                                            topKOp.sort(), topKOp.element_type());

    return mlir::success();
}

}  // namespace

void vpux::IE::TopKOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
