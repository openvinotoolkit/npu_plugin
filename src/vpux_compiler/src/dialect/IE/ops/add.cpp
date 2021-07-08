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
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

namespace {

//
// ConvertAddToScale
//

class ConvertAddToScale final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    using mlir::OpRewritePattern<IE::AddOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::AddOp biasOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertAddToScale::matchAndRewrite(IE::AddOp biasOp, mlir::PatternRewriter& rewriter) const {
    static const auto N = Dim(0);
    static const auto H = Dim(2);
    static const auto W = Dim(3);

    auto mulOutShape = getShape(biasOp.output());
    auto biasesShape = getShape(biasOp.input2());

    if (mulOutShape.size() != 4 || biasesShape.size() != 4) {
        return mlir::failure();
    }
    if (biasesShape[N] != 1 || biasesShape[H] != 1 || biasesShape[W] != 1) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::ScaleShiftOp>(biasOp, biasOp.getType(), biasOp.input1(), nullptr, biasOp.input2());

    return mlir::success();
}

}  // namespace

mlir::LogicalResult vpux::IE::AddOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::AddOpAdaptor add(operands, attrs);
    if (mlir::failed(add.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = add.input1().getType().cast<mlir::ShapedType>();
    const auto in2Type = add.input2().getType().cast<mlir::ShapedType>();

    const auto outShapeRes =
            IE::broadcastEltwiseShape(in1Type.getShape(), in2Type.getShape(), add.auto_broadcast().getValue(), loc);
    if (mlir::succeeded(outShapeRes)) {
        inferredReturnShapes.emplace_back(outShapeRes.getValue(), in1Type.getElementType());
    }

    return outShapeRes;
}

void vpux::IE::AddOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.insert<ConvertAddToScale>(context);
}

mlir::OpFoldResult vpux::IE::AddOp::fold(ArrayRef<mlir::Attribute> operands) {
    VPUX_THROW_UNLESS(operands.size() == 2, "Wrong number of operands : {0}", operands.size());

    if (const auto attr = operands[1].dyn_cast_or_null<Const::ContentAttr>()) {
        const auto content = attr.fold();
        if (content.isSplat() && content.getSplatValue<float>() == 0.0f) {
            return input1();
        }
    }

    return nullptr;
}
