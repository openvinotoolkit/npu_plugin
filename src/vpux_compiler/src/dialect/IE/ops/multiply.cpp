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
#include "vpux/compiler/dialect/IE/utils/to_ngraph.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/numeric.hpp"

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
    static const auto C = Dim(1);
    static const auto H = Dim(2);
    static const auto W = Dim(3);

    const auto lhsType = mulOp.input1().getType().cast<mlir::ShapedType>();
    const auto outShapeRes = mulOp.output().getType().cast<mlir::ShapedType>();

    bool lhsIsActivation = (lhsType == outShapeRes);
    auto activationInput = lhsIsActivation ? mulOp.input1() : mulOp.input2();
    auto weightsInput = lhsIsActivation ? mulOp.input2() : mulOp.input1();

    auto mulOutShape = getShape(mulOp.output());
    auto weightsShape = getShape(weightsInput);

    if (mulOutShape.size() != 4 || weightsShape.size() != 4) {
        return mlir::failure();
    }
    if (weightsShape[N] != 1 || weightsShape[H] != 1 || weightsShape[W] != 1) {
        return mlir::failure();
    }

    if (weightsShape[C] != mulOutShape[C] && weightsShape[C] != 1) {
        return mlir::failure();
    }

    // Broadcast scalar for all channels
    if (weightsShape[C] != mulOutShape[C] && weightsShape[C] == 1) {
        auto input2Const = weightsInput.getDefiningOp<Const::DeclareOp>();
        if (input2Const == nullptr) {
            return mlir::failure();
        }
        Const::ContentAttr dataAttr = input2Const.contentAttr().broadcast(C, mulOutShape[C]);

        if (dataAttr == nullptr) {
            return mlir::failure();
        }

        auto dataConstOp = rewriter.create<Const::DeclareOp>(mulOp.getLoc(), dataAttr.getType(), dataAttr);

        weightsInput = dataConstOp.output();
    }

    rewriter.replaceOpWithNewOp<IE::ScaleShiftOp>(mulOp, mulOp.getType(), activationInput, weightsInput, nullptr);
    return mlir::success();
}

}  // namespace

mlir::LogicalResult vpux::IE::MultiplyOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
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
        if (content.isSplat() && isDoubleEqual(content.getSplatValue<double>(), 1.0)) {
            return input1();
        }
    }

    return nullptr;
}

std::unique_ptr<ngraph::Node> vpux::IE::MultiplyOp::toNgraph(ngraph::OutputVector &outputs)
{
    const ngraph::op::AutoBroadcastType autoBroadCastType = exportBroadcastType(auto_broadcast());

    return std::make_unique<opset_latest::Multiply>(outputs.at(0), outputs.at(1),
        ngraph::op::AutoBroadcastSpec(autoBroadCastType));
}
