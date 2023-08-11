//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

//
// verify
//

mlir::LogicalResult vpux::IE::ClampOp::verify() {
    auto inElemType = input().getType().cast<vpux::NDTypeInterface>().getElementType();
    if (inElemType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        return errorAt(*this, "Per-axis quantized type is not supported. Got: {0}", inElemType);
    }

    return mlir::success();
}

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::ClampOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ClampOpAdaptor clamp(operands, attrs);
    if (mlir::failed(clamp.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = clamp.input().getType().cast<mlir::ShapedType>();
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

//
// inferElemTypeInfo
//

void vpux::IE::ClampOp::inferElemTypeInfo(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    const auto inputElemType = info.getInput(0);

    if (inputElemType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        // Do not propagate element type down in per channel case.
        return;
    }

    for (size_t outputInd = 0; outputInd < info.getNumOutputs(); ++outputInd) {
        info.setOutput(outputInd, inputElemType);
    }
}

void vpux::IE::ClampOp::inferElemTypeInfoUp(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    const auto outputElemType = info.getOutput(0);

    if (outputElemType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        // Do not propagate element type up in per channel case.
        return;
    }

    for (size_t inputInd = 0; inputInd < info.getNumInputs(); ++inputInd) {
        info.setInput(inputInd, outputElemType);
    }
}

//
// Fuse Clamps
//

namespace {
class FuseClamps final : public mlir::OpRewritePattern<IE::ClampOp> {
public:
    using mlir::OpRewritePattern<IE::ClampOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ClampOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseClamps::matchAndRewrite(IE::ClampOp origOp, mlir::PatternRewriter& rewriter) const {
    auto parentOp = origOp.input().getDefiningOp<IE::ClampOp>();
    if (parentOp == nullptr) {
        return mlir::failure();
    }

    if (!parentOp.getResult().hasOneUse()) {
        return mlir::failure();
    }

    const auto minParentOp = parentOp.minAttr().getValueAsDouble();
    const auto minOrigOp = origOp.minAttr().getValueAsDouble();
    const auto maxParentOp = parentOp.maxAttr().getValueAsDouble();
    const auto maxOrigOp = origOp.maxAttr().getValueAsDouble();

    const auto newMin = std::max(minParentOp, minOrigOp);
    const auto newMax = std::min(maxParentOp, maxOrigOp);

    rewriter.replaceOpWithNewOp<IE::ClampOp>(origOp, parentOp.input(), getFPAttr(rewriter, newMin),
                                             getFPAttr(rewriter, newMax));
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::ClampOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<FuseClamps>(ctx);
}
