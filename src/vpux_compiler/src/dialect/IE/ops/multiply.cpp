//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// FuseMulAndAdd
//

namespace {

class FuseMulAndAdd final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    using mlir::OpRewritePattern<IE::AddOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::AddOp biasOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseMulAndAdd::matchAndRewrite(IE::AddOp biasOp, mlir::PatternRewriter& rewriter) const {
    static const auto N = Dim(0);
    static const auto C = Dim(1);
    static const auto H = Dim(2);
    static const auto W = Dim(3);

    auto mulOp = mlir::dyn_cast_or_null<IE::MultiplyOp>(biasOp.input1().getDefiningOp());
    if (mulOp == nullptr) {
        return mlir::failure();
    }

    auto mulOutShape = getShape(mulOp.output());
    auto weightsShape = getShape(mulOp.input2());
    auto biasShape = getShape(biasOp.input2());

    if (mulOutShape.size() != 4 || biasShape.size() != 4 || weightsShape.size() != 4) {
        return mlir::failure();
    }
    if (biasShape[N] != 1 || biasShape[H] != 1 || biasShape[W] != 1) {
        return mlir::failure();
    }
    if (weightsShape[N] != 1 || weightsShape[H] != 1 || weightsShape[W] != 1) {
        return mlir::failure();
    }
    if (biasShape[C] != weightsShape[C]) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::ScaleShiftOp>(biasOp, biasOp.getType(), mulOp.input1(), mulOp.input2(),
                                                  biasOp.input2());

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

void vpux::IE::MultiplyOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns,
                                                       mlir::MLIRContext* context) {
    patterns.insert<FuseMulAndAdd>(context);
}

mlir::OpFoldResult vpux::IE::MultiplyOp::fold(ArrayRef<mlir::Attribute> operands) {
    VPUX_THROW_UNLESS(operands.size() == 2, "Wrong number of operands : {0}", operands.size());

    if (const auto cst = operands[1].dyn_cast_or_null<ConstContentAttr>()) {
        if (cst.isSplat() && cst.getSplatValue<float>() == 1.0f) {
            return input1();
        }
    }

    return nullptr;
}