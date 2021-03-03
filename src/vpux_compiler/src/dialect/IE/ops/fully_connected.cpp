//
// Copyright 2021 Intel Corporation.
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

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::FullyConnectedOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::FullyConnectedOpAdaptor fullyConnected(operands, attrs);
    if (mlir::failed(fullyConnected.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = fullyConnected.input().getType().cast<mlir::ShapedType>();
    const auto weightsType = fullyConnected.weights().getType().cast<mlir::ShapedType>();
    const auto inShape = inType.getShape();
    const auto weightsShape = weightsType.getShape();
    const auto inRank = inShape.size();
    const auto weightsRank = weightsShape.size();

    if (weightsRank != 2 || inRank != 2) {
        return mlir::failure();
    }

    SmallVector<int64_t> outShape;
    outShape.push_back(inShape[0]);
    outShape.push_back(weightsShape[0]);

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}

//
// FuseFCAndBias
//

namespace {

class FuseFCAndBias final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    using mlir::OpRewritePattern<IE::AddOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::AddOp biasOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseFCAndBias::matchAndRewrite(IE::AddOp biasOp, mlir::PatternRewriter& rewriter) const {
    static const auto N = Dim(0);
    static const auto C = Dim(1);

    if (!biasOp.input1().hasOneUse()) {
        return mlir::failure();
    }

    auto fullyConnectedOp = mlir::dyn_cast_or_null<IE::FullyConnectedOp>(biasOp.input1().getDefiningOp());
    if (fullyConnectedOp == nullptr) {
        return mlir::failure();
    }

    if (fullyConnectedOp.bias() != nullptr) {
        return mlir::failure();
    }

    auto fcOutShape = getShape(fullyConnectedOp.output());
    auto biasShape = getShape(biasOp.input2());

    if (fcOutShape.size() != 2 || biasShape.size() != 2) {
        return mlir::failure();
    }
    if (biasShape[N] != 1) {
        return mlir::failure();
    }
    if (biasShape[C] != fcOutShape[C]) {
        return mlir::failure();
    }

    auto* newFC = rewriter.clone(*fullyConnectedOp);
    newFC->insertOperands(newFC->getNumOperands(), biasOp.input2());

    rewriter.replaceOp(biasOp, newFC->getOpResults());

    return mlir::success();
}

}  // namespace

void vpux::IE::FullyConnectedOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns,
                                                             mlir::MLIRContext* context) {
    patterns.insert<FuseFCAndBias>(context);
}
