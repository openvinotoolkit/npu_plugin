//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ReverseSequenceOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ReverseSequenceOpAdaptor rev(operands, attrs);
    if (mlir::failed(rev.verify(loc))) {
        return mlir::failure();
    }

    const auto dataType = rev.data().getType().cast<mlir::ShapedType>();
    const auto dataShape = dataType.getShape();

    if (dataShape.size() < 2) {
        return errorAt(loc, "First input tensor's size should not be less than 2D. Got {0}D tensor", dataShape.size());
    }

    const auto seqShape = getShape(rev.seq_length());

    if (seqShape.size() != 1) {
        return errorAt(loc, "Second input tensor should be 1D Tensor. Got {0}D tensor", seqShape.size());
    }

    const auto dataDims = checked_cast<int64_t>(dataShape.size());

    const auto batch_axis = rev.batch_axis();

    if (batch_axis >= dataDims || batch_axis < -dataDims) {
        return errorAt(loc, "ReverseSequence Parameter batch axis {0} out of the tensor rank range [{1}, {2}].",
                       batch_axis, -dataDims, dataDims - 1);
    }

    const auto seq_axis = rev.seq_axis();

    if (seq_axis >= dataDims || seq_axis < -dataDims) {
        return errorAt(loc, "ReverseSequence Parameter sequence axis {0} out of the tensor rank range [{1}, {2}].",
                       seq_axis, -dataDims, dataDims - 1);
    }

    if (seqShape[Dims4D::Act::N] != dataShape[batch_axis]) {
        return errorAt(loc, "Sequence lengths input size {0} is not equal to batch axis dimension of data input {1}",
                       seqShape[Dims4D::Act::N], dataShape[batch_axis]);
    }

    const auto elementType = dataType.getElementType();
    if (!(elementType.isF16() || elementType.isF32() || elementType.isUnsignedInteger(8))) {
        return errorAt(loc, "Reverse Sequence only support FP16, FP32, U8 data type");
    }

    inferredReturnShapes.emplace_back(dataShape, elementType);

    return mlir::success();
}

mlir::OpFoldResult vpux::IE::ReverseSequenceOp::fold(ArrayRef<mlir::Attribute> operands) {
    VPUX_THROW_UNLESS(operands.size() == 2, "Wrong number of operands : {0}", operands.size());

    if (const auto attr = operands[1].dyn_cast_or_null<Const::ContentAttr>()) {
        const auto content = attr.fold();
        if (content.isSplat() && content.getSplatValue<int32_t>() == 1) {
            return data();
        }
    }

    return nullptr;
}

namespace {
class ConvertU8ToFP16 final : public mlir::OpRewritePattern<IE::ReverseSequenceOp> {
public:
    using mlir::OpRewritePattern<IE::ReverseSequenceOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ReverseSequenceOp rsOp, mlir::PatternRewriter& rewriter) const final;
};

class NormalizeAxis final : public mlir::OpRewritePattern<IE::ReverseSequenceOp> {
public:
    using mlir::OpRewritePattern<IE::ReverseSequenceOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ReverseSequenceOp rsOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertU8ToFP16::matchAndRewrite(IE::ReverseSequenceOp rsOp,
                                                     mlir::PatternRewriter& rewriter) const {
    const auto dataType = rsOp.data().getType().cast<mlir::ShapedType>();

    if (dataType.getElementType().isUnsignedInteger(8)) {
        auto convertOpBefore =
                rewriter.create<IE::ConvertOp>(rsOp.getLoc(), rsOp.data(), mlir::Float16Type::get(getContext()));
        auto reverseSequenceOp = rewriter.create<IE::ReverseSequenceOp>(
                rsOp.getLoc(), convertOpBefore.output(), rsOp.seq_length(), rsOp.seq_axis(), rsOp.batch_axis());
        auto inputTypeAttr = mlir::TypeAttr::get(
                mlir::IntegerType::get(getContext(), 8, mlir::IntegerType::SignednessSemantics::Unsigned));
        rewriter.replaceOpWithNewOp<IE::ConvertOp>(rsOp, reverseSequenceOp.output(), inputTypeAttr);
        return mlir::success();
    }
    return mlir::failure();
}

mlir::LogicalResult NormalizeAxis::matchAndRewrite(IE::ReverseSequenceOp rsOp, mlir::PatternRewriter& rewriter) const {
    const auto dataShape = getShape(rsOp.data());
    const auto dataDims = checked_cast<int64_t>(dataShape.size());
    const auto seq_axis = rsOp.seq_axis();
    const auto batch_axis = rsOp.batch_axis();
    if (seq_axis >= 0 && batch_axis >= 0) {
        return mlir::failure();
    }
    const auto normalized_seq_axis = seq_axis >= 0 ? seq_axis : seq_axis + dataDims;
    const auto normalized_batch_axis = batch_axis >= 0 ? batch_axis : batch_axis + dataDims;
    rewriter.replaceOpWithNewOp<IE::ReverseSequenceOp>(rsOp, rsOp.data(), rsOp.seq_length(), normalized_seq_axis,
                                                       normalized_batch_axis);
    return mlir::success();
}

}  // namespace

void vpux::IE::ReverseSequenceOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                              mlir::MLIRContext* context) {
    patterns.add<NormalizeAxis>(context);
    patterns.add<ConvertU8ToFP16>(context);
}
