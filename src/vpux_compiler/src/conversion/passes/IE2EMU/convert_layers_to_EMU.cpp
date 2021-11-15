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

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

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
    rewriter.replaceOpWithNewOp<EMU::SplitUPAOp>(origOp, origOp.getResults().getTypes(), origOp.input(),
                                                 origOp.axis_valueAttr(), origOp.num_splitsAttr());
    _log.trace("Replaced with 'EMU.SplitUPAOp'");

    return mlir::success();
}

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
    auto resultTypes = origOp.getResults().getTypes();
    rewriter.replaceOpWithNewOp<EMU::CTCGreedyDecoderSeqLenUPAOp>(origOp, resultTypes[0], resultTypes[1],
                                                                  origOp.input(), origOp.sequenceLength(),
                                                                  origOp.blankIndex(), origOp.mergeRepeatedAttr());
    _log.trace("Replaced with 'EMU.CTCGreedyDecoderSeqLenOp'");

    return mlir::success();
}

//
// LSTMCellRewrite
//

class LSTMCellRewrite final : public mlir::OpRewritePattern<IE::LSTMCellOp> {
public:
    LSTMCellRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::LSTMCellOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LSTMCellOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LSTMCellRewrite::matchAndRewrite(IE::LSTMCellOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found LSTMCell Operation '{0}'", origOp->getLoc());
    auto resultTypes = origOp.getResults().getTypes();
    rewriter.replaceOpWithNewOp<EMU::LSTMCellUPAOp>(origOp, resultTypes[0], resultTypes[1], origOp.inputData(),
                                                    origOp.initialHiddenState(), origOp.initialCellState(),
                                                    origOp.weights(), origOp.biases());
    _log.trace("Replaced with 'EMU.LSTMCellOp'");

    return mlir::success();
}

//
// LSTMSequenceRewrite
//

class LSTMSequenceRewrite final : public mlir::OpRewritePattern<IE::LSTMSequenceOp> {
public:
    LSTMSequenceRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::LSTMSequenceOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LSTMSequenceOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LSTMSequenceRewrite::matchAndRewrite(IE::LSTMSequenceOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("Found LSTMSequence Operation '{0}'", origOp->getLoc());
    auto resultTypes = origOp.getResults().getTypes();
    rewriter.replaceOpWithNewOp<EMU::LSTMSequenceUPAOp>(origOp, resultTypes[0], resultTypes[1], resultTypes[2],
                                                        origOp.inputData(), origOp.initialHiddenState(),
                                                        origOp.initialCellState(), origOp.weights(), origOp.biases(),
                                                        origOp.sequenceLengthAttr(), origOp.directionAttr());
    _log.trace("Replaced with 'EMU.LSTMSequenceOp'");

    return mlir::success();
}

//
// FakeQuantizeRewrite
//

class FakeQuantizeRewrite final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    FakeQuantizeRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FakeQuantizeRewrite::matchAndRewrite(IE::FakeQuantizeOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("Found FakeQuantize Operation '{0}'", origOp->getLoc());

    auto inLowConst = origOp.input_low().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = origOp.input_high().getDefiningOp<Const::DeclareOp>();
    auto outLowConst = origOp.output_low().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = origOp.output_high().getDefiningOp<Const::DeclareOp>();

    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        return matchFailed(rewriter, origOp, "Got non constant parameters");
    }

    rewriter.replaceOpWithNewOp<EMU::FakeQuantizeUPAOp>(
            origOp, origOp.output().getType(), origOp.input(), origOp.levels(), inLowConst.contentAttr(),
            inHighConst.contentAttr(), outLowConst.contentAttr(), outHighConst.contentAttr());
    _log.trace("Replaced with 'EMU.FakeQuantizeUPAOp'");
    return mlir::success();
}

// Bias reshape

mlir::FailureOr<mlir::Value> reshapeBias(mlir::PatternRewriter& rewriter, mlir::Value bias) {
    auto biasConst = bias.getDefiningOp<Const::DeclareOp>();
    if (biasConst == nullptr) {
        return mlir::failure();
    }
    const auto origBiasType = bias.getType().cast<mlir::ShapedType>();
    const auto origBiasShape = origBiasType.getShape();

    const auto newBiasShape = ShapeRef({origBiasShape[1]});
    const auto newBiasType = changeShape(origBiasType, newBiasShape);
    const auto newBiasConstAttr = biasConst.contentAttr().reshape(newBiasShape);
    return rewriter.replaceOpWithNewOp<Const::DeclareOp>(biasConst, newBiasType, newBiasConstAttr).output();
}

//
// FullyConnectedRewrite
//

class FullyConnectedRewrite final : public mlir::OpRewritePattern<IE::FullyConnectedOp> {
public:
    FullyConnectedRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FullyConnectedOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FullyConnectedOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FullyConnectedRewrite::matchAndRewrite(IE::FullyConnectedOp origOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Found FullyConnected Operation '{0}'", origOp->getLoc());

    if (origOp.bias() == nullptr) {
        rewriter.replaceOpWithNewOp<EMU::FullyConnectedUPAOp>(origOp, origOp.output().getType(), origOp.input(),
                                                              origOp.weights(), nullptr);
        return mlir::success();
    }

    auto newBias = reshapeBias(rewriter, origOp.bias());
    if (mlir::failed(newBias)) {
        return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<EMU::FullyConnectedUPAOp>(origOp, origOp.output().getType(), origOp.input(),
                                                          origOp.weights(), newBias.getValue());
    _log.trace("Replaced with 'EMU.FullyConnectedUPAOp'");
    return mlir::success();
}

//
// ConvolutionRewrite
//

class ConvolutionRewrite final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    ConvolutionRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvolutionRewrite::matchAndRewrite(IE::ConvolutionOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Convolution Operation '{0}'", origOp->getLoc());

    const int64_t groups = 1;
    if (origOp.bias() == nullptr) {
        rewriter.replaceOpWithNewOp<EMU::ConvolutionUPAOp>(
                origOp, origOp.output().getType(), origOp.input(), origOp.filter(), nullptr, origOp.strides(),
                origOp.dilations(), origOp.pads_begin(), origOp.pads_end(), groups);
        return mlir::success();
    }

    auto newBias = reshapeBias(rewriter, origOp.bias());
    if (mlir::failed(newBias)) {
        return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<EMU::ConvolutionUPAOp>(
            origOp, origOp.output().getType(), origOp.input(), origOp.filter(), newBias.getValue(), origOp.strides(),
            origOp.dilations(), origOp.pads_begin(), origOp.pads_end(), groups);
    _log.trace("Replaced with 'EMU.ConvolutionUPAOp'");
    return mlir::success();
}

//
// Generated
//

#include <vpux/compiler/conversion/rewriters/generated/convert_layers_to_EMU.hpp.inc>

//
// ConvertLayers2EMUPass
//

class ConvertLayers2EMUPass final : public ConvertLayers2EMUBase<ConvertLayers2EMUPass> {
public:
    explicit ConvertLayers2EMUPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertLayers2EMUPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalDialect<IE::IEDialect>();
    target.addIllegalDialect<IERT::IERTDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<EMU::EMUDialect>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<SplitRewrite>(&ctx, _log);
    patterns.insert<CTCGreedyDecoderSeqLenRewrite>(&ctx, _log);
    patterns.insert<LSTMCellRewrite>(&ctx, _log);
    patterns.insert<LSTMSequenceRewrite>(&ctx, _log);
    patterns.insert<FakeQuantizeRewrite>(&ctx, _log);
    patterns.insert<FullyConnectedRewrite>(&ctx, _log);
    patterns.insert<ConvolutionRewrite>(&ctx, _log);
    populateWithGenerated(patterns);

    auto func = getFunction();
    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertLayers2EMUPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertLayers2EMUPass(Logger log) {
    return std::make_unique<ConvertLayers2EMUPass>(log);
}
