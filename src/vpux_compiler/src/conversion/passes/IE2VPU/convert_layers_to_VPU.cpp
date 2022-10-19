//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// CTCGreedyDecoderSeqLenRewrite
//

class CTCGreedyDecoderSeqLenRewrite final : public mlir::OpRewritePattern<IE::CTCGreedyDecoderSeqLenOp> {
public:
    CTCGreedyDecoderSeqLenRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::CTCGreedyDecoderSeqLenOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::CTCGreedyDecoderSeqLenOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult CTCGreedyDecoderSeqLenRewrite::matchAndRewrite(IE::CTCGreedyDecoderSeqLenOp origOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("Found CTCGreedyDecoderSeqLen Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::CTCGreedyDecoderSeqLenOp>(origOp, origOp.input(), origOp.sequenceLength(),
                                                               origOp.blankIndex(), origOp.mergeRepeatedAttr());

    return mlir::success();
}


//
// ProposalRewrite
//

class ProposalRewrite final : public mlir::OpRewritePattern<IE::ProposalOp> {
public:
    ProposalRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ProposalOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ProposalOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ProposalRewrite::matchAndRewrite(IE::ProposalOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Proposal Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::ProposalOp>(origOp, origOp.class_probs(), origOp.bbox_deltas(),
                                                 origOp.image_shape(), origOp.proposal_attrsAttr());
    _log.trace("Replaced with 'VPU.ProposalOp'");

    return mlir::success();
}

//
// SplitRewrite
//

class SplitRewrite final : public mlir::OpRewritePattern<IE::SplitOp> {
public:
    SplitRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SplitOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SplitOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SplitRewrite::matchAndRewrite(IE::SplitOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Split Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::SplitOp>(origOp, origOp.input(), origOp.axis(), origOp.num_splitsAttr(),
                                              origOp.axis_valueAttr());

    return mlir::success();
}

//
// Generated
//

#include <vpux/compiler/conversion/rewriters/generated/convert_layers_to_VPU.hpp.inc>

//
// ConvertLayers2VPUPass
//

class ConvertLayers2VPUPass final : public ConvertLayers2VPUBase<ConvertLayers2VPUPass> {
public:
    explicit ConvertLayers2VPUPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertLayers2VPUPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalDialect<IE::IEDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPU::VPUDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<CTCGreedyDecoderSeqLenRewrite>(&ctx, _log);
    patterns.insert<ProposalRewrite>(&ctx, _log);
    patterns.insert<SplitRewrite>(&ctx, _log);
    populateWithGenerated(patterns);

    auto func = getFunction();
    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertLayers2VPUPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertLayers2VPUPass(Logger log) {
    return std::make_unique<ConvertLayers2VPUPass>(log);
}
