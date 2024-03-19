//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/passes/IE2VPU/convert_layers_to_VPU.hpp"
#include "vpux/compiler/VPU30XX/conversion.hpp"
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
    rewriter.replaceOpWithNewOp<VPU::EmbeddingSegmentsSumOp>(
            origOp, origOp.getEmbTable(), /*indices=*/nullptr, /*segment_ids=*/nullptr, /*per_sample_weights=*/nullptr,
            origOp.getIndicesValueAttr(), origOp.getSegmentIdsValueAttr(), origOp.getNumSegmentsValueAttr(),
            origOp.getDefaultIndexValueAttr(), origOp.getPerSampleWeightsValueAttr());
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
    rewriter.replaceOpWithNewOp<VPU::EmbeddingBagOffsetsSumOp>(
            origOp, origOp.getEmbTable(), /*indices=*/nullptr, /*offsets=*/nullptr, /*per_sample_weights=*/nullptr,
            origOp.getIndicesValueAttr(), origOp.getOffsetsValueAttr(), origOp.getDefaultIndexValueAttr(),
            origOp.getPerSampleWeightsValueAttr());
    return mlir::success();
}

//
// Generated
//

#include <vpux/compiler/conversion/convert_layers_to_VPU.hpp.inc>

//
// ConvertLayers2VPUPass
//

class ConvertLayers2VPUPass final : public arch30xx::ConvertLayers2VPUBase<ConvertLayers2VPUPass> {
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

std::unique_ptr<mlir::Pass> vpux::arch30xx::createConvertLayers2VPUPass(Logger log) {
    return std::make_unique<ConvertLayers2VPUPass>(log);
}
