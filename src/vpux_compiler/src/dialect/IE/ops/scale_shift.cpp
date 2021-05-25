//
// Copyright 2021 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
// add

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"

using namespace vpux;

#include <mlir/IR/PatternMatch.h>

namespace {

//
// FuseScaleAndBias
//

class FuseScaleAndBias final : public mlir::OpRewritePattern<IE::ScaleShiftOp> {
public:
    using mlir::OpRewritePattern<IE::ScaleShiftOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ScaleShiftOp biasOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseScaleAndBias::matchAndRewrite(IE::ScaleShiftOp biasOp, mlir::PatternRewriter& rewriter) const {
    static const auto C = Dim(1);

    if (!biasOp.input().hasOneUse()) {
        return mlir::failure();
    }

    if (biasOp.weights() != nullptr) {
        return mlir::failure();
    }

    auto scaleOp = mlir::dyn_cast_or_null<IE::ScaleShiftOp>(biasOp.input().getDefiningOp());
    if (scaleOp == nullptr || scaleOp.biases() != nullptr) {
        return mlir::failure();
    }

    auto mulOutShape = getShape(scaleOp.output());
    auto weightsShape = getShape(scaleOp.weights());
    auto biasShape = getShape(biasOp.biases());

    if (mulOutShape.size() != 4) {
        return mlir::failure();
    }
    if (biasShape[C] != weightsShape[C]) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::ScaleShiftOp>(biasOp, biasOp.getType(), scaleOp.input(), scaleOp.weights(),
                                                  biasOp.biases());

    return mlir::success();
}

}  // namespace

mlir::LogicalResult vpux::IE::ScaleShiftOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ScaleShiftOpAdaptor scaleShift(operands, attrs);
    if (mlir::failed(scaleShift.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = scaleShift.input().getType().cast<mlir::ShapedType>();
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

void vpux::IE::ScaleShiftOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                         mlir::MLIRContext* context) {
    patterns.insert<FuseScaleAndBias>(context);
}
