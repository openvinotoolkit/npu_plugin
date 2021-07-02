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
#include "vpux/compiler/dialect/const/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

namespace {

//
// ConvertMultiplyToScale
//

class ConvertMultiplyToScale final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    using mlir::OpRewritePattern<IE::MultiplyOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp mulOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertMultiplyToScale::matchAndRewrite(IE::MultiplyOp mulOp,
                                                            mlir::PatternRewriter& rewriter) const {
    static const auto N = Dim(0);
    static const auto H = Dim(2);
    static const auto W = Dim(3);

    auto mulOutShape = getShape(mulOp.output());
    auto weightsShape = getShape(mulOp.input2());

    if (mulOutShape.size() != 4 || weightsShape.size() != 4) {
        return mlir::failure();
    }
    if (weightsShape[N] != 1 || weightsShape[H] != 1 || weightsShape[W] != 1) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::ScaleShiftOp>(mulOp, mulOp.getType(), mulOp.input1(), mulOp.input2(), nullptr);

    return mlir::success();
}

}  // namespace

mlir::LogicalResult vpux::IE::MultiplyOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::MultiplyOpAdaptor multiply(operands, attrs);
    if (mlir::failed(multiply.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = multiply.input1().getType().cast<mlir::ShapedType>();
    const auto in2Type = multiply.input2().getType().cast<mlir::ShapedType>();

    const auto outShapeRes = IE::broadcastEltwiseShape(in1Type.getShape(), in2Type.getShape(),
                                                       multiply.auto_broadcast().getValue(), loc);

    if (mlir::succeeded(outShapeRes)) {
        inferredReturnShapes.emplace_back(outShapeRes.getValue(), in1Type.getElementType());
    }

    return mlir::success();
}

void vpux::IE::MultiplyOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.insert<ConvertMultiplyToScale>(context);
}

mlir::OpFoldResult vpux::IE::MultiplyOp::fold(ArrayRef<mlir::Attribute> operands) {
    VPUX_THROW_UNLESS(operands.size() == 2, "Wrong number of operands : {0}", operands.size());

    if (const auto attr = operands[1].dyn_cast_or_null<Const::ContentAttr>()) {
        const auto content = attr.fold();
        if (content.isSplat() && content.getSplatValue<float>() == 1.0f) {
            return input1();
        }
    }

    return nullptr;
}
