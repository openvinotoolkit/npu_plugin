//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/groupconvolution_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <ngraph/coordinate_diff.hpp>
#include <ngraph/op/op.hpp>

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// ConvertGroupConvToConvPass
//

class ConvertGroupConvToConvPass final : public IE::ConvertGroupConvToConvBase<ConvertGroupConvToConvPass> {
public:
    explicit ConvertGroupConvToConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class GroupConvolutionOpConverter;

private:
    void safeRunOnFunc() final;
};

//
// GroupConvolutionOpConverter
//

class ConvertGroupConvToConvPass::GroupConvolutionOpConverter final :
        public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    GroupConvolutionOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertGroupConvToConvPass::GroupConvolutionOpConverter::matchAndRewrite(
        IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got GroupConvolutionOp layer at '{0}'", origOp->getLoc());
    VPUX_THROW_UNLESS(origOp.getType().getRank() == 4, "The pass currently can only support 4D input");

    const auto input = origOp.input();
    const auto inputShape = input.getType().cast<vpux::NDTypeInterface>().getShape();
    const auto weights = origOp.filter();
    const auto weightsShape = weights.getType().cast<vpux::NDTypeInterface>().getShape();
    const auto bias = origOp.bias();
    const auto group = origOp.groups().value();
    const auto newInShape = Shape{inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C] / group,
                                  inputShape[Dims4D::Act::H], inputShape[Dims4D::Act::W]};
    const auto inputShapeAttr = getIntArrayAttr(getContext(), newInShape);
    const auto newWeightsShape = Shape{weightsShape[Dims4D::Filter::OC] / group, weightsShape[Dims4D::Filter::IC],
                                       weightsShape[Dims4D::Filter::KY], weightsShape[Dims4D::Filter::KX]};
    const auto weightsShapeAttr = getIntArrayAttr(getContext(), newWeightsShape);

    SmallVector<mlir::Value> slices;
    mlir::Value biasSlice;
    mlir::Value weightsSlice;
    for (const auto sliceIdx : irange(group)) {
        // Slice input
        Shape inputOffsets = Shape(inputShape.size(), 0);
        inputOffsets[Dims4D::Act::C] = checked_cast<int64_t>(inputShape[Dims4D::Act::C] / group * sliceIdx);
        const auto inputOffsetsAttr = getIntArrayAttr(getContext(), inputOffsets);
        const auto inputSlice =
                rewriter.createOrFold<IE::SliceOp>(origOp->getLoc(), input, inputOffsetsAttr, inputShapeAttr);

        // Slice weights
        Shape weightsOffsets = Shape(weightsShape.size(), 0);
        weightsOffsets[Dims4D::Filter::OC] = checked_cast<int64_t>(weightsShape[Dims4D::Filter::OC] / group * sliceIdx);
        const auto weightsOffsetsAttr = getIntArrayAttr(getContext(), weightsOffsets);
        auto fakeQuantizeOp = weights.getDefiningOp<IE::FakeQuantizeOp>();
        if (fakeQuantizeOp != nullptr) {
            const auto newFakeQuantizeParamShape = Shape{weightsShape[Dims4D::Filter::OC] / group, 1, 1, 1};
            const auto fakeQuantizeParamShapeAttr = getIntArrayAttr(getContext(), newFakeQuantizeParamShape);
            auto inputLow = fakeQuantizeOp.input_low();
            auto inputHigh = fakeQuantizeOp.input_high();
            auto outputLow = fakeQuantizeOp.output_low();
            auto outputHigh = fakeQuantizeOp.output_high();

            auto newInput = rewriter.createOrFold<IE::SliceOp>(fakeQuantizeOp->getLoc(), fakeQuantizeOp.input(),
                                                               weightsOffsetsAttr, weightsShapeAttr);
            if (inputLow.getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Filter::OC] != 1) {
                inputLow = rewriter.createOrFold<IE::SliceOp>(fakeQuantizeOp->getLoc(), inputLow, weightsOffsetsAttr,
                                                              fakeQuantizeParamShapeAttr);
            }
            if (outputLow.getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Filter::OC] != 1) {
                outputLow = rewriter.createOrFold<IE::SliceOp>(fakeQuantizeOp->getLoc(), outputLow, weightsOffsetsAttr,
                                                               fakeQuantizeParamShapeAttr);
            }
            if (inputHigh.getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Filter::OC] != 1) {
                inputHigh = rewriter.createOrFold<IE::SliceOp>(fakeQuantizeOp->getLoc(), inputHigh, weightsOffsetsAttr,
                                                               fakeQuantizeParamShapeAttr);
            }
            if (outputHigh.getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Filter::OC] != 1) {
                outputHigh = rewriter.createOrFold<IE::SliceOp>(fakeQuantizeOp->getLoc(), outputHigh,
                                                                weightsOffsetsAttr, fakeQuantizeParamShapeAttr);
            }

            weightsSlice = rewriter.create<IE::FakeQuantizeOp>(fakeQuantizeOp.getLoc(), newInput, inputLow, inputHigh,
                                                               outputLow, outputHigh, fakeQuantizeOp.levels(),
                                                               fakeQuantizeOp.auto_broadcast());
        } else {
            weightsSlice =
                    rewriter.createOrFold<IE::SliceOp>(origOp->getLoc(), weights, weightsOffsetsAttr, weightsShapeAttr);
        }

        // Slice Bias
        if (bias != nullptr) {
            auto biasShape = bias.getType().cast<vpux::NDTypeInterface>().getShape();
            const auto newBiasShape = Shape{biasShape[Dims4D::Act::N], biasShape[Dims4D::Act::C] / group,
                                            biasShape[Dims4D::Act::H], biasShape[Dims4D::Act::W]};
            const auto biasShapeAttr = getIntArrayAttr(getContext(), newBiasShape);
            Shape biasOffsets = Shape(biasShape.size(), 0);
            biasOffsets[Dims4D::Act::C] = checked_cast<int64_t>(newBiasShape[Dims4D::Act::C] * sliceIdx);
            const auto biasOffsetsAttr = getIntArrayAttr(getContext(), biasOffsets);
            biasSlice = rewriter.createOrFold<IE::SliceOp>(origOp->getLoc(), bias, biasOffsetsAttr, biasShapeAttr);
        } else {
            biasSlice = nullptr;
        }

        // New conv
        auto newConvLoc = appendLoc(origOp->getLoc(), "_ConvertGroupConv_{0}", sliceIdx);
        auto convOp =
                rewriter.create<IE::ConvolutionOp>(newConvLoc, inputSlice, weightsSlice, biasSlice, origOp.strides(),
                                                   origOp.pads_begin(), origOp.pads_end(), origOp.dilations(), nullptr);

        slices.push_back(convOp);
    }

    rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, slices, Dims4D::Act::C.ind());

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertGroupConvToConvPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::ConversionTarget target(ctx);

    target.addDynamicallyLegalOp<IE::GroupConvolutionOp>([&](IE::GroupConvolutionOp op) {
        return mlir::failed(IE::canConvertGroupConvToConv(op));
    });
    target.addLegalOp<IE::ConvolutionOp>();
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::FakeQuantizeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GroupConvolutionOpConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertGroupConvToConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertGroupConvToConvPass(Logger log) {
    return std::make_unique<ConvertGroupConvToConvPass>(log);
}
