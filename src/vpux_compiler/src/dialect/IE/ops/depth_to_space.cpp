//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"

using namespace vpux;

void vpux::IE::DepthToSpaceOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                     int64_t block_size, IE::DepthToSpaceMode mode) {
    build(builder, state, input, block_size, mode, nullptr);
}

void vpux::IE::DepthToSpaceOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                     mlir::IntegerAttr block_size, IE::DepthToSpaceModeAttr mode) {
    build(builder, state, input, block_size, mode, nullptr);
}

void vpux::IE::DepthToSpaceOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type outType,
                                     mlir::Value input, mlir::IntegerAttr block_size, IE::DepthToSpaceModeAttr mode) {
    build(builder, state, outType, input, block_size, mode, nullptr);
}

mlir::LogicalResult vpux::IE::DepthToSpaceOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::DepthToSpaceOpAdaptor depthToSpace(operands, attrs);
    if (mlir::failed(depthToSpace.verify(loc))) {
        return mlir::failure();
    }

    const auto inShape = getShape(depthToSpace.input());
    const auto inType = depthToSpace.input().getType().cast<mlir::ShapedType>().getElementType();
    const auto block_size = depthToSpace.block_size();
    auto paddedChannels = depthToSpace.padded_channels();

    if (!(inType.isF16() || inType.isF32() || inType.isUnsignedInteger(8) ||
          inType.isa<mlir::quant::QuantizedType>())) {
        return errorAt(loc, "DepthToSpace only support FP16, FP32, U8, quant data type");
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

    int64_t paddedIC = 0;
    int64_t paddedOC = 0;

    auto blockSizeSquare = block_size * block_size;
    if (paddedChannels.hasValue()) {
        paddedIC = paddedChannels.getValue().input() ? paddedChannels.getValue().input().getInt() : 0;
        paddedOC = paddedChannels.getValue().output() ? paddedChannels.getValue().output().getInt() : 0;

        auto unpaddedChannels = inShape[Dims4D::Act::C] - paddedIC;
        if (unpaddedChannels % blockSizeSquare != 0) {
            return errorAt(loc, "Invalid block size {0}, which is not divisible by input shape {1}", block_size,
                           unpaddedChannels);
        }

        if (paddedOC != 0 &&
            (inShape[Dims4D::Act::C] / blockSizeSquare != unpaddedChannels / blockSizeSquare + paddedOC)) {
            return errorAt(loc, "Invalid padded output channels {0}", paddedOC);
        }
    }

    size_t W_out = inShape[Dims4D::Act::W] * block_size;
    size_t H_out = inShape[Dims4D::Act::H] * block_size;
    size_t C_out = (inShape[Dims4D::Act::C] - paddedIC) / blockSizeSquare + paddedOC;
    size_t N_out = inShape[Dims4D::Act::N];

    SmallVector<int64_t> outShape{checked_cast<int64_t>(N_out), checked_cast<int64_t>(C_out),
                                  checked_cast<int64_t>(H_out), checked_cast<int64_t>(W_out)};

    const auto inputType = depthToSpace.input().getType().cast<vpux::NDTypeInterface>();
    const auto outDesc = IE::getTensorAttr(ctx, inputType.getDimsOrder(), inputType.getMemSpace());
    inferredReturnShapes.emplace_back(outShape, inType, outDesc);
    return mlir::success();
}

//
// inferLayoutInfo
//

// After support layout with NCHW convert to DMA, can remove this function
// This is ticket E#41656 to support NCHW layout
void vpux::IE::DepthToSpaceOp::inferLayoutInfo(vpux::IE::LayerLayoutInfo& info) {
    info.setInput(0, DimsOrder::NHWC);
    info.setOutput(0, DimsOrder::NHWC);
}

namespace {
class ConvertInputToFP16 final : public mlir::OpRewritePattern<IE::DepthToSpaceOp> {
public:
    using mlir::OpRewritePattern<IE::DepthToSpaceOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::DepthToSpaceOp Op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertInputToFP16::matchAndRewrite(IE::DepthToSpaceOp op, mlir::PatternRewriter& rewriter) const {
    const auto module = op->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);
    const auto inputType = op.input().getType().cast<mlir::ShapedType>().getElementType();
    if (arch != VPU::ArchKind::VPUX37XX && inputType.isUnsignedInteger(8)) {
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
    patterns.add<ConvertInputToFP16>(context);
}

mlir::OpFoldResult vpux::IE::DepthToSpaceOp::fold(ArrayRef<mlir::Attribute> operands) {
    VPUX_THROW_UNLESS(operands.size() == 1, "Wrong number of operands : {0}", operands.size());
    // when block_size == 1, fold to input itself
    if (block_size() == 1) {
        return input();
    }

    return nullptr;
}
