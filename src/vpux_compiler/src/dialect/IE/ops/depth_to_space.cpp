//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::DepthToSpaceOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::DepthToSpaceOpAdaptor depthToSpace(operands, attrs);
    if (mlir::failed(depthToSpace.verify(loc))) {
        return mlir::failure();
    }

    const auto inShape = getShape(depthToSpace.input());
    const auto inType = depthToSpace.input().getType().cast<mlir::ShapedType>().getElementType();
    const auto block_size = depthToSpace.block_size().getInt();

    if (!(inType.isF16() || inType.isF32() || inType.isUnsignedInteger(8))) {
        return errorAt(loc, "DepthToSpace only support FP16, FP32, U8 data type");
    }

    if (inShape.size() < 3) {
        return errorAt(loc, "Invalid input tensor shape, dimension must be greater than 2.");
    }

    if (block_size <= 0) {
        return errorAt(loc, "Invalid block size {0}, should be greater than zero", block_size);
    }

    if (inShape[Dims4D::Act::C] % (block_size * block_size) != 0) {
        return errorAt(loc, "Invalid block size {0}, which is not divisible by input shape {1}", block_size,
                       inShape[Dims4D::Act::C]);
    }

    size_t W_out = inShape[Dims4D::Act::W] * block_size;
    size_t H_out = inShape[Dims4D::Act::H] * block_size;
    size_t C_out = inShape[Dims4D::Act::C] / (block_size * block_size);
    size_t N_out = inShape[Dims4D::Act::N];

    SmallVector<int64_t> outShape{checked_cast<int64_t>(N_out), checked_cast<int64_t>(C_out),
                                  checked_cast<int64_t>(H_out), checked_cast<int64_t>(W_out)};

    inferredReturnShapes.emplace_back(outShape, inType);
    return mlir::success();
}

namespace {
class ConvertInputToFP16 final : public mlir::OpRewritePattern<IE::DepthToSpaceOp> {
public:
    using mlir::OpRewritePattern<IE::DepthToSpaceOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::DepthToSpaceOp Op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertInputToFP16::matchAndRewrite(IE::DepthToSpaceOp op, mlir::PatternRewriter& rewriter) const {
    const auto inputType = op.input().getType().cast<mlir::ShapedType>().getElementType();
    if (inputType.isUnsignedInteger(8)) {
        auto convertOpBefore =
                rewriter.create<IE::ConvertOp>(op.getLoc(), op.input(), mlir::Float16Type::get(getContext()));
        auto depthtospaceOp =
                rewriter.create<IE::DepthToSpaceOp>(op.getLoc(), convertOpBefore.output(), op.block_size(), op.mode());

        rewriter.replaceOpWithNewOp<IE::ConvertOp>(op, depthtospaceOp.output(), inputType);
        return mlir::success();
    }
    return mlir::failure();
}

}  // namespace

void vpux::IE::DepthToSpaceOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                           mlir::MLIRContext* context) {
    patterns.insert<ConvertInputToFP16>(context);
}

mlir::OpFoldResult vpux::IE::DepthToSpaceOp::fold(ArrayRef<mlir::Attribute> operands) {
    VPUX_THROW_UNLESS(operands.size() == 1, "Wrong number of operands : {0}", operands.size());
    // when block_size == 1, fold to input itself
    if (block_size() == 1) {
        return input();
    }

    return nullptr;
}
