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
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"

using namespace vpux;

namespace {

//
// ConvertToFP16
//

class ConvertToFP16 final : public mlir::OpRewritePattern<IE::NotEqualOp> {
public:
    using mlir::OpRewritePattern<IE::NotEqualOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::NotEqualOp notEqualOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertToFP16::matchAndRewrite(IE::NotEqualOp notEqualOp, mlir::PatternRewriter& rewriter) const {
    const auto in1Type = notEqualOp.input1().getType().cast<mlir::ShapedType>();
    const auto outType = notEqualOp.output().getType().cast<mlir::ShapedType>();

    if (!(in1Type.getElementType().isF16())) {
        auto float16Type = mlir::Float16Type::get(getContext());
        auto convertIn1 = rewriter.create<IE::ConvertOp>(notEqualOp.getLoc(), notEqualOp.input1(), float16Type);
        auto convertIn2 = rewriter.create<IE::ConvertOp>(notEqualOp.getLoc(), notEqualOp.input2(), float16Type);

        auto newNotEqualOp = rewriter.create<IE::NotEqualOp>(notEqualOp.getLoc(), convertIn1.output(),
                                                             convertIn2.output(), notEqualOp.auto_broadcastAttr());

        rewriter.replaceOpWithNewOp<IE::ConvertOp>(notEqualOp, newNotEqualOp.output(), outType.getElementType());
        return mlir::success();
    } else {
        return mlir::success();
    }
}

}  // namespace

mlir::LogicalResult vpux::IE::NotEqualOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::NotEqualOpAdaptor notEqual(operands, attrs);
    if (mlir::failed(notEqual.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = notEqual.input1().getType().cast<mlir::ShapedType>();
    const auto in2Type = notEqual.input2().getType().cast<mlir::ShapedType>();

    const auto outShapeRes = IE::broadcastEltwiseShape(in1Type.getShape(), in2Type.getShape(),
                                                       notEqual.auto_broadcast().getValue(), loc);
    if (mlir::succeeded(outShapeRes)) {
        inferredReturnShapes.emplace_back(outShapeRes.getValue(), in1Type.getElementType());
    }

    return outShapeRes;
}

void vpux::IE::NotEqualOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.insert<ConvertToFP16>(context);
}
