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

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Transforms/DialectConversion.h>
#include <ngraph/slice_plan.hpp>

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

    auto inLowConst = origOp.input_low().getDefiningOp<ConstantInterface>();
    auto inHighConst = origOp.input_high().getDefiningOp<ConstantInterface>();
    auto outLowConst = origOp.output_low().getDefiningOp<ConstantInterface>();
    auto outHighConst = origOp.output_high().getDefiningOp<ConstantInterface>();

    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        return matchFailed(rewriter, origOp, "Got non constant parameters");
    }

    rewriter.replaceOpWithNewOp<VPUIP::FakeQuantizeUPAOp>(origOp, origOp.input(), origOp.output_buff(), origOp.levels(),
                                                          inLowConst.getContent(), inHighConst.getContent(),
                                                          outLowConst.getContent(), outHighConst.getContent());

    return mlir::success();
}

//
// StridedSliceRewrite
//

class StridedSliceRewrite final : public mlir::OpRewritePattern<IERT::StridedSliceOp> {
public:
    StridedSliceRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IERT::StridedSliceOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::StridedSliceOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult StridedSliceRewrite::matchAndRewrite(IERT::StridedSliceOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("Found StridedSlice Operation '{0}'", origOp->getLoc());

    const auto getAxisSetArr = [](mlir::ArrayAttr attr) {
        ngraph::AxisSet axis_set;

        const auto arr = parseIntArrayAttr(attr);
        for (const auto& p : arr | indexed) {
            if (p.value() == 1) {
                axis_set.emplace(p.index());
            }
        }

        return axis_set;
    };

    const auto beginsVec = to_std_vector(parseIntArrayAttr(origOp.begins_attr()));
    const auto endsVec = to_std_vector(parseIntArrayAttr(origOp.ends_attr()));
    const auto stridesVec = to_std_vector(parseIntArrayAttr(origOp.strides_attr()));

    const auto beginMask = getAxisSetArr(origOp.begin_mask());
    const auto endMask = getAxisSetArr(origOp.end_mask());
    const auto newAxisMask = getAxisSetArr(origOp.new_axis_mask());
    const auto shrinkAxisMask = getAxisSetArr(origOp.shrink_axis_mask());
    const auto ellipsisMask = getAxisSetArr(origOp.ellipsis_mask());

    const auto inDataType = origOp.input().getType().cast<mlir::ShapedType>();
    const auto inDataShape = inDataType.getShape();

    const auto plan =
            ngraph::make_slice_plan(ngraph::Shape(inDataShape.begin(), inDataShape.end()), beginsVec, endsVec,
                                    stridesVec, beginMask, endMask, newAxisMask, shrinkAxisMask, ellipsisMask);

    const auto ctx = rewriter.getContext();
    const auto beginsAttr = getInt32ArrayAttr(ctx, plan.begins);
    const auto endsAttr = getInt32ArrayAttr(ctx, plan.ends);
    const auto stridesAttr = getInt32ArrayAttr(ctx, plan.strides);

    rewriter.replaceOpWithNewOp<VPUIP::StridedSliceUPAOp>(origOp, origOp.input(), origOp.output_buff(), beginsAttr,
                                                          endsAttr, stridesAttr);

    _log.trace("Replaced with 'VPUIP.StridedSliceOp'");

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
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();
    target.addLegalOp<IERT::ConstantOp, IERT::StaticAllocOp>();
    target.addLegalOp<IERT::GenericReshapeOp, IERT::ConcatViewOp, mlir::memref::SubViewOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<CTCGreedyDecoderSeqLenRewrite>(&ctx, _log);
    patterns.insert<FakeQuantizeRewrite>(&ctx, _log);
    patterns.insert<StridedSliceRewrite>(&ctx, _log);
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
