//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::GenericReshapeOp::verify() {
    const auto op = getOperation();
    auto distributedInType = input().getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto distributedOutType = output().getType().dyn_cast<VPUIP::DistributedBufferType>();
    if (distributedInType && distributedOutType) {
        if (!isCompatibleForDistributedInputOutput(op, distributedInType, distributedOutType)) {
            return errorAt(op, "Reshape input and output must have the same distribution mode");
        }
        if (!isDistributedCompatibleAfterShapeChange(distributedInType, distributedOutType.getShape(),
                                                     VPU::getArch(getOperation()))) {
            return errorAt(op, "Reshape has incompatible output shape as clustering");
        }
    }

    const auto inType = input().getType().cast<vpux::NDTypeInterface>();
    const auto outType = output().getType().cast<vpux::NDTypeInterface>();

    if (inType.getNumElements() != outType.getNumElements()) {
        return errorAt(op, "Reshape input and output must have the same number of elements");
    }

    const auto inReqs = StrideReqs::compact(inType.getRank());
    const auto outReqs = StrideReqs::compact(outType.getRank());

    if (!inReqs.checkStrides(inType)) {
        return errorAt(op, "Input strides do not match requirements '{0}'", inType);
    }
    if (!outReqs.checkStrides(outType)) {
        return errorAt(op, "Output strides do not match requirements '{0}'", inType);
    }

    return mlir::success();
}

mlir::Value VPUIP::GenericReshapeOp::getViewSource() {
    return input();
}

mlir::OpFoldResult VPUIP::GenericReshapeOp::fold(ArrayRef<mlir::Attribute> operands) {
    if (input().getType() == output().getType()) {
        return input();
    }

    if (const auto cst = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        return cst.reshape(getShape(output()));
    }

    return nullptr;
}

//
// FuseReshapes
//

namespace {

class FuseReshapes final : public mlir::OpRewritePattern<VPUIP::GenericReshapeOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPUIP::GenericReshapeOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseReshapes::matchAndRewrite(VPUIP::GenericReshapeOp origOp,
                                                  mlir::PatternRewriter& rewriter) const {
    auto producerReshapeOp = origOp.input().getDefiningOp<VPUIP::GenericReshapeOp>();
    if (producerReshapeOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPUIP::GenericReshapeOp>(origOp, origOp.output().getType(), producerReshapeOp.input());

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void VPUIP::GenericReshapeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<FuseReshapes>(ctx);
}
