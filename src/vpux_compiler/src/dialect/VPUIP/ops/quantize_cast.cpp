//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"

using namespace vpux;

mlir::Value VPUIP::QuantizeCastOp::getViewSource() {
    return input();
}

mlir::OpFoldResult VPUIP::QuantizeCastOp::fold(ArrayRef<mlir::Attribute>) {
    return input().getType() == output().getType() ? input() : nullptr;
}

//
// FuseQuantizeCastOps
//

namespace {

class FuseQuantizeCastOps final : public mlir::OpRewritePattern<VPUIP::QuantizeCastOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPUIP::QuantizeCastOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseQuantizeCastOps::matchAndRewrite(VPUIP::QuantizeCastOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    auto producerQuantizeCastOp = origOp.input().getDefiningOp<VPUIP::QuantizeCastOp>();
    if (producerQuantizeCastOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPUIP::QuantizeCastOp>(origOp, origOp.getType(), producerQuantizeCastOp.input());

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void VPUIP::QuantizeCastOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<FuseQuantizeCastOps>(ctx);
}

mlir::LogicalResult vpux::VPUIP::QuantizeCastOp::verify() {
    const auto op = getOperation();
    auto distributedInType = input().getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto distributedOutType = output().getType().dyn_cast<VPUIP::DistributedBufferType>();
    if (distributedInType && distributedOutType) {
        auto inputDistribution = distributedInType.getDistribution();
        auto outputDistribution = distributedOutType.getDistribution();
        if (inputDistribution != outputDistribution) {
            return errorAt(op, "QuantizeCastOp input and output must have the same distribution attribute");
        }
    }

    auto inElemType = input().getType().cast<NDTypeInterface>().getElementType();
    auto outElemType = output().getType().cast<NDTypeInterface>().getElementType();
    if (inElemType.isa<mlir::FloatType>() || outElemType.isa<mlir::FloatType>()) {
        return errorAt(op, " QuantizeCastOp input and output must have the legal element type");
    }

    auto inputStrides = getStrides(input());
    auto outputStrides = getStrides(output());
    if (inputStrides != outputStrides) {
        return errorAt(op, "QuantizeCastOp input and output must have the same strides, but got {0} and {1}",
                       inputStrides, outputStrides);
    }

    auto inputShape = getShape(input());
    auto outputShape = getShape(output());
    if (inputShape != outputShape) {
        return errorAt(op, "QuantizeCastOp input and output must have the same shape, but got {0} and {1}", inputShape,
                       outputShape);
    }
    return mlir::success();
}
