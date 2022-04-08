//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::SpaceToDepthOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::SpaceToDepthOpAdaptor spd(operands, attrs);
    if (mlir::failed(spd.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = spd.input().getType().cast<mlir::ShapedType>();

    const auto elementType = inputType.getElementType();
    if (!(elementType.isF16() || elementType.isF32() || elementType.isUnsignedInteger(8))) {
        return errorAt(loc, "SpaceToDepth only supports FP16, FP32, U8 input data type");
    }

    const auto inputShape = inputType.getShape();
    const auto block_size = spd.block_size().getInt();

    if (inputShape.size() < 3) {
        return errorAt(loc, "Input tensor rank must be greater than 2. Got {0}D tensor", inputShape.size());
    }

    if (block_size <= 0) {
        return errorAt(loc, "Invalid block size {0}, should be greater than zero", block_size);
    }

    static const auto N = Dims4D::Act::N;
    static const auto C = Dims4D::Act::C;
    static const auto H = Dims4D::Act::H;
    static const auto W = Dims4D::Act::W;

    if (inputShape[H.ind()] % block_size || inputShape[W.ind()] % block_size) {
        return errorAt(loc, "Invalid block_size {0} , height {1} and width {2} must be divisible by block_size",
                       block_size, inputShape[H.ind()], inputShape[W.ind()]);
    }

    const auto outN = inputShape[N.ind()];
    const auto outC = inputShape[C.ind()] * block_size * block_size;
    const auto outH = inputShape[H.ind()] / block_size;
    const auto outW = inputShape[W.ind()] / block_size;

    SmallVector<int64_t> outShape{outN, outC, outH, outW};
    inferredReturnShapes.emplace_back(outShape, inputType.getElementType());

    return mlir::success();
}

namespace {
class ConvertInputToFP16 final : public mlir::OpRewritePattern<IE::SpaceToDepthOp> {
public:
    using mlir::OpRewritePattern<IE::SpaceToDepthOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::SpaceToDepthOp Op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertInputToFP16::matchAndRewrite(IE::SpaceToDepthOp op, mlir::PatternRewriter& rewriter) const {
    const auto inputType = op.input().getType().cast<mlir::ShapedType>().getElementType();
    if (inputType.isUnsignedInteger(8)) {
        auto convertOpBefore =
                rewriter.create<IE::ConvertOp>(op.getLoc(), op.input(), mlir::Float16Type::get(getContext()));
        auto spacetodepthOp =
                rewriter.create<IE::SpaceToDepthOp>(op.getLoc(), convertOpBefore.output(), op.block_size(), op.mode());
        rewriter.replaceOpWithNewOp<IE::ConvertOp>(op, spacetodepthOp.output(), inputType);
        return mlir::success();
    }
    return mlir::failure();
}

}  // namespace

void vpux::IE::SpaceToDepthOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                           mlir::MLIRContext* context) {
    patterns.insert<ConvertInputToFP16>(context);
}

mlir::OpFoldResult vpux::IE::SpaceToDepthOp::fold(ArrayRef<mlir::Attribute> operands) {
    VPUX_THROW_UNLESS(operands.size() == 1, "Wrong number of operands : {0}", operands.size());

    if (block_size() == 1) {
        return input();
    }

    return nullptr;
}
