//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::PReluOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::PReluOpAdaptor prelu(operands, attrs);
    if (mlir::failed(prelu.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = prelu.input().getType().cast<mlir::ShapedType>();
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

mlir::LogicalResult vpux::IE::LeakyReluOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::LeakyReluOpAdaptor leaky_relu(operands, attrs);
    if (mlir::failed(leaky_relu.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = leaky_relu.input().getType().cast<mlir::ShapedType>();
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

namespace {

class UseLeakyRelu final : public mlir::OpRewritePattern<IE::PReluOp> {
public:
    using mlir::OpRewritePattern<IE::PReluOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::PReluOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult UseLeakyRelu::matchAndRewrite(IE::PReluOp origOp, mlir::PatternRewriter& rewriter) const {
    auto negativeSlopeOp = origOp.negative_slope().getDefiningOp<IE::ConstantOp>();
    if (negativeSlopeOp == nullptr) {
        return mlir::failure();
    }

    auto negativeSlopeContent = negativeSlopeOp.getContent();
    if (!negativeSlopeContent.isSplat()) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::LeakyReluOp>(origOp, origOp.getType(), origOp.input(),
                                                 rewriter.getF32FloatAttr(negativeSlopeContent.getSplatValue<float>()));

    return mlir::success();
}

}  // namespace

void vpux::IE::PReluOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.insert<UseLeakyRelu>(context);
}
