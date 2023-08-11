//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::BatchNormInferenceOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::BatchNormInferenceOpAdaptor norm(operands, attrs);
    if (mlir::failed(norm.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = norm.input().getType().cast<mlir::ShapedType>();
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::BatchNormInferenceOp> {
public:
    using mlir::OpRewritePattern<IE::BatchNormInferenceOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::BatchNormInferenceOp normOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::BatchNormInferenceOp normOp,
                                                        mlir::PatternRewriter& rewriter) const {
    auto gamma = normOp.gamma();
    auto beta = normOp.beta();
    auto mean = normOp.mean();
    auto var = normOp.variance();

    if ((gamma == nullptr) || (beta == nullptr) || (mean == nullptr) || (var == nullptr)) {
        // already converted
        return mlir::failure();
    }

    auto gammaConst = gamma.getDefiningOp<Const::DeclareOp>();
    auto betaConst = beta.getDefiningOp<Const::DeclareOp>();
    auto meanConst = mean.getDefiningOp<Const::DeclareOp>();
    auto varConst = var.getDefiningOp<Const::DeclareOp>();

    // fail if not constants
    if ((gammaConst == nullptr) || (betaConst == nullptr) || (meanConst == nullptr) || (varConst == nullptr)) {
        return mlir::failure();
    }

    const auto gammaContent = gammaConst.content();
    const auto betaContent = betaConst.content();
    const auto meanContent = meanConst.content();
    const auto varContent = varConst.content();

    const auto gammaAttr = getFPArrayAttr(getContext(), gammaContent.getValues<double>());
    const auto betaAttr = getFPArrayAttr(getContext(), betaContent.getValues<double>());
    const auto meanAttr = getFPArrayAttr(getContext(), meanContent.getValues<double>());
    const auto varAttr = getFPArrayAttr(getContext(), varContent.getValues<double>());

    rewriter.replaceOpWithNewOp<IE::BatchNormInferenceOp>(normOp, normOp.getType(), normOp.input(), nullptr, nullptr,
                                                          nullptr, nullptr, gammaAttr, betaAttr, meanAttr, varAttr,
                                                          normOp.eps());
    return mlir::success();
}

}  // namespace

void vpux::IE::BatchNormInferenceOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                                 mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
