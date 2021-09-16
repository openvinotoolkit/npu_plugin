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
// ConvertSubtractToAddandNegative
//

class ConvertSubtractToAddandNegative final : public mlir::OpRewritePattern<IE::SubtractOp> {
public:
    using mlir::OpRewritePattern<IE::SubtractOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::SubtractOp subOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertSubtractToAddandNegative::matchAndRewrite(IE::SubtractOp subOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    auto input1 = subOp.input1();
    auto input2 = subOp.input2();
    auto inputType = subOp.input2().getType();
    auto loc = subOp.getLoc();
    auto negativeOp = rewriter.create<IE::NegativeOp>(loc, inputType, input2);
    auto NegativeInput2 = negativeOp.output();

    rewriter.replaceOpWithNewOp<IE::AddOp>(subOp, input1, NegativeInput2, subOp.auto_broadcastAttr(),
                                           /*post_op=*/nullptr, /*clip_op=*/nullptr);
    return mlir::success();
}

}  // namespace

mlir::LogicalResult vpux::IE::SubtractOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::SubtractOpAdaptor subtract(operands, attrs);
    if (mlir::failed(subtract.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = subtract.input1().getType().cast<mlir::ShapedType>();
    const auto in2Type = subtract.input2().getType().cast<mlir::ShapedType>();

    const auto outShapeRes = IE::broadcastEltwiseShape(in1Type.getShape(), in2Type.getShape(),
                                                       subtract.auto_broadcast().getValue(), loc);
    if (mlir::succeeded(outShapeRes)) {
        inferredReturnShapes.emplace_back(outShapeRes.getValue(), in1Type.getElementType());
    }

    return outShapeRes;
}

void vpux::IE::SubtractOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.insert<ConvertSubtractToAddandNegative>(context);
}

mlir::OpFoldResult vpux::IE::SubtractOp::fold(ArrayRef<mlir::Attribute> operands) {
    VPUX_THROW_UNLESS(operands.size() == 2, "Wrong number of operands : {0}", operands.size());

    if (const auto attr = operands[1].dyn_cast_or_null<Const::ContentAttr>()) {
        const auto content = attr.fold();
        if (content.isSplat() && content.getSplatValue<float>() == 0.0f) {
            return input1();
        }
    }

    return nullptr;
}
