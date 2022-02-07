//
// Copyright 2020 Intel Corporation.
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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

namespace {

//
// UseBaseBuffers
//

class UseBaseBuffers final : public mlir::OpRewritePattern<VPUIP::WeightsTableOp> {
public:
    using mlir::OpRewritePattern<VPUIP::WeightsTableOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::WeightsTableOp wtOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::Value retrieveBaseBuffer(mlir::Value val) {
    if (val == nullptr) {
        return val;
    }

    if (auto layerOp = val.getDefiningOp<IERT::LayerOpInterface>()) {
        VPUX_THROW_WHEN(layerOp == nullptr, "Value '{0}' was produced by non layer operation", val.getLoc());

        const auto opRes = val.cast<mlir::OpResult>();
        const auto resInd = opRes.getResultNumber();

        return layerOp.getOutputs()[resInd];
    }

    return val;
}

mlir::LogicalResult UseBaseBuffers::matchAndRewrite(VPUIP::WeightsTableOp wtOp, mlir::PatternRewriter& rewriter) const {
    const auto newOpInput = retrieveBaseBuffer(wtOp.op_input());
    const auto newOpOutput = retrieveBaseBuffer(wtOp.op_output());
    const auto newWeights = retrieveBaseBuffer(wtOp.weights());
    const auto newActWindow = retrieveBaseBuffer(wtOp.activation_window());

    if (newOpInput == wtOp.op_input() && newOpOutput == wtOp.op_output() && newWeights == wtOp.weights() &&
        newActWindow == wtOp.activation_window()) {
        return matchFailed(rewriter, wtOp, "The operation is already in canonical form");
    }

    rewriter.replaceOpWithNewOp<VPUIP::WeightsTableOp>(wtOp, wtOp.getType(), newOpInput, newOpOutput, newWeights,
                                                       newActWindow, wtOp.biasAttr(), wtOp.ppeAttr());
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::VPUIP::WeightsTableOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                              mlir::MLIRContext* ctx) {
    patterns.add<UseBaseBuffers>(ctx);
}
