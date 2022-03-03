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

#include "vpux/compiler/dialect/VPUIP/utils.hpp"
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

class CTCGreedyDecoderSeqLenRewrite final : public mlir::OpRewritePattern<IERT::CTCGreedyDecoderSeqLenOp> {
public:
    CTCGreedyDecoderSeqLenRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IERT::CTCGreedyDecoderSeqLenOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::CTCGreedyDecoderSeqLenOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult CTCGreedyDecoderSeqLenRewrite::matchAndRewrite(IERT::CTCGreedyDecoderSeqLenOp origOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("Found CTCGreedyDecoderSeqLen Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPUIP::CTCGreedyDecoderSeqLenUPAOp>(
            origOp, origOp.input(), origOp.sequenceLength(), origOp.blankIndex(), origOp.output_buff(),
            origOp.outputLength_buff(), origOp.mergeRepeatedAttr());
    _log.trace("Replaced with 'VPUIP.CTCGreedyDecoderSeqLenOp'");

    return mlir::success();
}

//
// LSTMCellRewrite
//

class LSTMCellRewrite final : public mlir::OpRewritePattern<IERT::LSTMCellOp> {
public:
    LSTMCellRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IERT::LSTMCellOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::LSTMCellOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LSTMCellRewrite::matchAndRewrite(IERT::LSTMCellOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found LSTMCell Operation '{0}'", origOp->getLoc());
    rewriter.replaceOpWithNewOp<VPUIP::LSTMCellUPAOp>(origOp, origOp.inputData(), origOp.initialHiddenState(),
                                                      origOp.initialCellState(), origOp.weights(), origOp.biases(),
                                                      origOp.outputHiddenState_buff(), origOp.outputCellState_buff());
    _log.trace("Replaced with 'VPUIP.LSTMCellOp'");

    return mlir::success();
}

//
// LSTMSequenceRewrite
//

class LSTMSequenceRewrite final : public mlir::OpRewritePattern<IERT::LSTMSequenceOp> {
public:
    LSTMSequenceRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IERT::LSTMSequenceOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::LSTMSequenceOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LSTMSequenceRewrite::matchAndRewrite(IERT::LSTMSequenceOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("Found LSTMSequence Operation '{0}'", origOp->getLoc());
    rewriter.replaceOpWithNewOp<VPUIP::LSTMSequenceUPAOp>(
            origOp, origOp.inputData(), origOp.initialHiddenState(), origOp.initialCellState(), origOp.weights(),
            origOp.biases(), origOp.outputHiddenValues_buff(), origOp.outputCellState_buff(),
            origOp.outputHiddenState_buff(), origOp.sequenceLengthAttr(), origOp.directionAttr());
    _log.trace("Replaced with 'VPUIP.LSTMSequenceOp'");

    return mlir::success();
}

//
// FakeQuantizeRewrite
//

class FakeQuantizeRewrite final : public mlir::OpRewritePattern<IERT::FakeQuantizeOp> {
public:
    FakeQuantizeRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IERT::FakeQuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FakeQuantizeRewrite::matchAndRewrite(IERT::FakeQuantizeOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("Found FakeQuantize Operation '{0}'", origOp->getLoc());

    auto inLowConst = origOp.input_low().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = origOp.input_high().getDefiningOp<Const::DeclareOp>();
    auto outLowConst = origOp.output_low().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = origOp.output_high().getDefiningOp<Const::DeclareOp>();

    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        return matchFailed(rewriter, origOp, "Got non constant parameters");
    }

    rewriter.replaceOpWithNewOp<VPUIP::FakeQuantizeUPAOp>(origOp, origOp.input(), origOp.output_buff(), origOp.levels(),
                                                          inLowConst.contentAttr(), inHighConst.contentAttr(),
                                                          outLowConst.contentAttr(), outHighConst.contentAttr());

    return mlir::success();
}

//
// FullyConnectedRewrite
//

class FullyConnectedRewrite final : public mlir::OpRewritePattern<IERT::FullyConnectedOp> {
public:
    FullyConnectedRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IERT::FullyConnectedOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::FullyConnectedOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FullyConnectedRewrite::matchAndRewrite(IERT::FullyConnectedOp origOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Found FullyConnected Operation '{0}'", origOp->getLoc());

    if (origOp.bias() == nullptr) {
        rewriter.replaceOpWithNewOp<VPUIP::FullyConnectedUPAOp>(origOp, origOp.input(), origOp.weights(), nullptr,
                                                                origOp.output_buff());
        return mlir::success();
    }

    const auto origBiasType = origOp.bias().getType().cast<vpux::NDTypeInterface>();

    const auto origBiasShape = origBiasType.getShape().raw();
    VPUX_THROW_UNLESS(origBiasShape[0] == 1, "Biases batch size is not equal 1");

    const std::array<int64_t, 1> newBiasShape = {origBiasShape[1]};
    const auto newBiasType = origBiasType.changeShape(ShapeRef(newBiasShape));

    auto newBias = rewriter.create<IERT::GenericReshapeOp>(origOp->getLoc(), newBiasType, origOp.bias());

    rewriter.replaceOpWithNewOp<VPUIP::FullyConnectedUPAOp>(origOp, origOp.input(), origOp.weights(), newBias.output(),
                                                            origOp.output_buff());

    return mlir::success();
}

//
// RewriteConvolution
//

class RewriteConvolution final : public mlir::OpRewritePattern<IERT::ConvolutionOp> {
public:
    RewriteConvolution(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IERT::ConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult RewriteConvolution::matchAndRewrite(IERT::ConvolutionOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Convolution Operation '{0}'", origOp->getLoc());

    const int64_t groups = 1;
    if (origOp.bias() == nullptr) {
        rewriter.replaceOpWithNewOp<VPUIP::ConvolutionUPAOp>(origOp, origOp.input(), origOp.filter(), nullptr,
                                                             origOp.output_buff(), origOp.strides(), origOp.dilations(),
                                                             origOp.pads_begin(), origOp.pads_end(), groups);
        return mlir::success();
    }

    const auto origBiasType = origOp.bias().getType().cast<vpux::NDTypeInterface>();
    const auto origBiasShape = origBiasType.getShape().raw();

    const std::array<int64_t, 1> newBiasShape = {origBiasShape[1]};
    const auto newBiasType = origBiasType.changeShape(ShapeRef(newBiasShape));
    auto newBias = rewriter.create<IERT::GenericReshapeOp>(origOp->getLoc(), newBiasType, origOp.bias());
    rewriter.replaceOpWithNewOp<VPUIP::ConvolutionUPAOp>(origOp, origOp.input(), origOp.filter(), newBias.output(),
                                                         origOp.output_buff(), origOp.strides(), origOp.dilations(),
                                                         origOp.pads_begin(), origOp.pads_end(), groups);
    return mlir::success();
}

//
// TimestampRewrite
//

class TimestampRewrite final : public mlir::OpRewritePattern<IERT::TimestampOp> {
public:
    TimestampRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IERT::TimestampOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::TimestampOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult TimestampRewrite::matchAndRewrite(IERT::TimestampOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Timestamp Operation '{0}'", origOp->getLoc());

    auto origType = origOp.getType().cast<vpux::NDTypeInterface>();
    VPUX_THROW_UNLESS(origType.getNumElements() == 1, "Got wrong elements number for TimestampOp");
    VPUX_THROW_UNLESS(origType.getElementType() == getUInt32Type(getContext()),
                      "Got wrong element type for TimestampOp");

    const auto timerType = origType.changeMemSpace(VPU::MemoryKind::Register);

    auto bufferOp = rewriter.create<VPURT::DeclareBufferOp>(origOp->getLoc(), timerType, VPURT::BufferSection::Register,
                                                            VPUIP::HW_TIMER_ABSOLUTE_ADDR);

    rewriter.replaceOpWithNewOp<VPUIP::NNDMAOp>(origOp, bufferOp.buffer(), origOp.output_buff());

    _log.trace("Replaced with 'VPURT::DeclareBufferOp'");

    return mlir::success();
}  // namespace

//
// TopKRewrite
//

class TopKRewrite final : public mlir::OpRewritePattern<IERT::TopKOp> {
public:
    TopKRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IERT::TopKOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::TopKOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult TopKRewrite::matchAndRewrite(IERT::TopKOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found TopK Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPUIP::TopKUPAOp>(origOp, origOp.input(), origOp.output_values_buff(),
                                                  origOp.target_shape_buff(), origOp.k_valueAttr(), origOp.axisAttr(),
                                                  origOp.modeAttr(), origOp.sortAttr(), origOp.element_typeAttr());

    return mlir::success();
}

//
// Generated
//

#include <vpux/compiler/conversion/rewriters/generated/convert_layers_to_VPUIP.hpp.inc>

//
// ConvertLayers2VPUIPPass
//

class ConvertLayers2VPUIPPass final : public ConvertLayers2VPUIPBase<ConvertLayers2VPUIPPass> {
public:
    explicit ConvertLayers2VPUIPPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertLayers2VPUIPPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalDialect<IERT::IERTDialect>();
    target.addLegalDialect<mlir::async::AsyncDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalDialect<VPURT::VPURTDialect>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();
    target.addLegalOp<Const::DeclareOp, IERT::StaticAllocOp>();
    target.addLegalOp<IERT::SubViewOp, IERT::ConcatViewOp>();
    target.addLegalOp<IERT::GenericReshapeOp, IERT::PermuteCastOp>();
    target.addLegalOp<IERT::QuantizeCastOp>();
    target.addLegalOp<VPUIP::SwKernelOp>();
    target.markOpRecursivelyLegal<VPUIP::SwKernelOp>([&](mlir::Operation*) {
        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<CTCGreedyDecoderSeqLenRewrite>(&ctx, _log);
    patterns.insert<LSTMCellRewrite>(&ctx, _log);
    patterns.insert<LSTMSequenceRewrite>(&ctx, _log);
    patterns.insert<FakeQuantizeRewrite>(&ctx, _log);
    patterns.insert<FullyConnectedRewrite>(&ctx, _log);
    patterns.insert<RewriteConvolution>(&ctx, _log);
    patterns.insert<TimestampRewrite>(&ctx, _log);
    patterns.insert<TopKRewrite>(&ctx, _log);
    populateWithGenerated(patterns);

    auto func = getFunction();
    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertLayers2VPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertLayers2VPUIPPass(Logger log) {
    return std::make_unique<ConvertLayers2VPUIPPass>(log);
}
