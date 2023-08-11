//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

namespace {
SmallVector<int64_t> dftOpsNormalizeAxes(mlir::Value input, SmallVector<int64_t> axes) {
    const auto inType = input.getType().cast<mlir::ShapedType>();
    const auto inRank = inType.getRank();
    for (size_t i = 0; i < axes.size(); ++i) {
        if (axes[i] < 0)
            axes[i] = inRank + axes[i];
    }
    return axes;
}
}  // namespace

mlir::LogicalResult vpux::IE::RDFTOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::RDFTOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = op.input().getType().cast<mlir::ShapedType>();
    auto outShape = to_small_vector(inType.getShape());
    SmallVector<int64_t> axes = dftOpsNormalizeAxes(op.input(), parseIntArrayAttr<int64_t>(op.axes_attr()));
    SmallVector<int64_t> signalSize = parseIntArrayAttr<int64_t>(op.signal_size_attr());

    for (size_t i = 0; i < axes.size(); ++i) {
        if (signalSize[i] != -1) {
            outShape[axes[i]] = signalSize[i];
        }
    }
    const auto lastAxis = axes.back();
    outShape[lastAxis] = outShape[lastAxis] / 2 + 1;
    outShape.push_back(2);
    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}

namespace {

//
// FtNormalizeAttributes
//

class FtNormalizeAttributes final : public mlir::OpRewritePattern<IE::RDFTOp> {
public:
    using mlir::OpRewritePattern<IE::RDFTOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::RDFTOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FtNormalizeAttributes::matchAndRewrite(IE::RDFTOp op, mlir::PatternRewriter& rewriter) const {
    if (!op.input()) {
        return mlir::failure();
    }
    SmallVector<int64_t> axes = dftOpsNormalizeAxes(op.input(), parseIntArrayAttr<int64_t>(op.axes_attr()));
    const auto axesAttr = getIntArrayAttr(getContext(), axes);
    if (axesAttr == op.axes_attr()) {
        return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<IE::RDFTOp>(op, op.input(), axesAttr, op.signal_size_attr());

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::RDFTOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<FtNormalizeAttributes>(context);
}
