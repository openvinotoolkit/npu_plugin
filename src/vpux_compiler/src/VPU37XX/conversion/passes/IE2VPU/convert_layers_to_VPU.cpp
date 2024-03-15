//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/passes/IE2VPU/convert_layers_to_VPU.hpp"
#include "vpux/compiler/VPU37XX/conversion.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// EmbeddingSegmentsSumRewriter
//

class EmbeddingSegmentsSumRewriter : public mlir::OpRewritePattern<IE::EmbeddingSegmentsSumOp> {
public:
    EmbeddingSegmentsSumRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::EmbeddingSegmentsSumOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::EmbeddingSegmentsSumOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EmbeddingSegmentsSumRewriter::matchAndRewrite(IE::EmbeddingSegmentsSumOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    const auto ctx = origOp->getContext();
    const auto weights = origOp.getPerSampleWeights();
    if (weights != nullptr) {
        rewriter.replaceOpWithNewOp<VPU::EmbeddingSegmentsSumOp>(
                origOp, origOp.getEmbTable(), origOp.getIndices(), origOp.getSegmentIds(), origOp.getPerSampleWeights(),
                /*indices_value=*/nullptr, /*segment_ids_value=*/nullptr, origOp.getNumSegmentsValueAttr(),
                origOp.getDefaultIndexValueAttr(), /*per_sample_weights_value=*/nullptr);
        return mlir::success();
    }

    mlir::RankedTensorType weightsTensorType;
    mlir::DenseElementsAttr baseAttr;
    const auto weightsShape = getShape(origOp.getIndices()).raw();
    const auto inType = origOp.getEmbTable().getType().cast<NDTypeInterface>();

    computeWeightForEmbeddingOp(ctx, weightsTensorType, baseAttr, weightsShape, inType);

    auto cstDeclOp =
            rewriter.create<Const::DeclareOp>(origOp.getLoc(), weightsTensorType, Const::ContentAttr::get(baseAttr));

    rewriter.replaceOpWithNewOp<VPU::EmbeddingSegmentsSumOp>(
            origOp, origOp.getEmbTable(), origOp.getIndices(), origOp.getSegmentIds(), cstDeclOp.getOutput(),
            /*indices_value=*/nullptr, /*segment_ids_value=*/nullptr, origOp.getNumSegmentsValueAttr(),
            origOp.getDefaultIndexValueAttr(), /*per_sample_weights_value=*/nullptr);
    return mlir::success();
}

//
// EmbeddingBagOffsetsSumRewriter
//

class EmbeddingBagOffsetsSumRewriter final : public mlir::OpRewritePattern<IE::EmbeddingBagOffsetsSumOp> {
public:
    EmbeddingBagOffsetsSumRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::EmbeddingBagOffsetsSumOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::EmbeddingBagOffsetsSumOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EmbeddingBagOffsetsSumRewriter::matchAndRewrite(IE::EmbeddingBagOffsetsSumOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("Found EmbeddingBagOffsetsSumOp Operation '{0}'", origOp->getLoc());

    const auto ctx = origOp->getContext();
    const auto weights = origOp.getPerSampleWeights();
    if (weights != nullptr) {
        rewriter.replaceOpWithNewOp<VPU::EmbeddingBagOffsetsSumOp>(
                origOp, origOp.getEmbTable(), origOp.getIndices(), origOp.getOffsets(), origOp.getPerSampleWeights(),
                /*indices_value=*/nullptr,
                /*offsets_value=*/nullptr, origOp.getDefaultIndexValueAttr(), /*per_sample_weights_value=*/nullptr);
        return mlir::success();
    }

    mlir::RankedTensorType weightsTensorType;
    mlir::DenseElementsAttr baseAttr;
    const auto weightsShape = getShape(origOp.getIndices()).raw();
    const auto inType = origOp.getEmbTable().getType().cast<NDTypeInterface>();

    computeWeightForEmbeddingOp(ctx, weightsTensorType, baseAttr, weightsShape, inType);

    auto cstDeclOp =
            rewriter.create<Const::DeclareOp>(origOp.getLoc(), weightsTensorType, Const::ContentAttr::get(baseAttr));

    rewriter.replaceOpWithNewOp<VPU::EmbeddingBagOffsetsSumOp>(
            origOp, origOp.getEmbTable(), origOp.getIndices(), origOp.getOffsets(), cstDeclOp.getOutput(),
            /*indices_value=*/nullptr, /*offsets_value=*/nullptr, origOp.getDefaultIndexValueAttr(),
            /*per_sample_weights_value=*/nullptr);

    return mlir::success();
}

//
// Generated
//

#include <vpux/compiler/conversion/convert_layers_to_VPU.hpp.inc>

//
// ConvertLayers2VPUPass
//

class ConvertLayers2VPUPass final : public arch37xx::ConvertLayers2VPUBase<ConvertLayers2VPUPass> {
public:
    explicit ConvertLayers2VPUPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertLayers2VPUPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::ConversionTarget target(ctx);
    target.addIllegalDialect<IE::IEDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPU::VPUDialect>();
    target.addLegalOp<mlir::func::FuncOp, mlir::func::ReturnOp, mlir::func::CallOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<IfRewrite>(&ctx, _log);
    patterns.add<CTCGreedyDecoderSeqLenRewrite>(&ctx, _log);
    patterns.add<ProposalRewrite>(&ctx, _log);
    patterns.add<SplitRewrite>(&ctx, _log);
    patterns.add<StubRewrite>(&ctx, _log);
    patterns.add<NonMaxSuppressionRewrite>(&ctx, _log);
    patterns.add<InterpolateRewrite>(&ctx, _log);
    patterns.add<EmbeddingSegmentsSumRewriter>(&ctx, _log);
    patterns.add<EmbeddingBagOffsetsSumRewriter>(&ctx, _log);
    patterns.add<GRUCellRewrite>(&ctx, _log);
    patterns.add<EmbeddingBagPackedSumRewrite>(&ctx, _log);
    patterns.add<TopKRewrite>(&ctx, _log);
    patterns.add<TransposedConvRewrite>(&ctx, _log);
    patterns.add<NormalizeL2Rewrite>(&ctx, _log);
    populateWithGenerated(patterns);

    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertLayers2VPUPass
//

std::unique_ptr<mlir::Pass> vpux::arch37xx::createConvertLayers2VPUPass(Logger log) {
    return std::make_unique<ConvertLayers2VPUPass>(log);
}
