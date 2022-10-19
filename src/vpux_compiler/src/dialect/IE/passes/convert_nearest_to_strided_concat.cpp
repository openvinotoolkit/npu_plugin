//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

mlir::Value createFQ(mlir::PatternRewriter& rewriter, mlir::Value input, IE::FakeQuantizeOp fq) {
    const auto outputType = fq.output().getType().cast<vpux::NDTypeInterface>();
    const auto newOutputType = outputType.changeShape(getShape(input));
    return rewriter
            .create<IE::FakeQuantizeOp>(fq.getLoc(), newOutputType, input, fq.input_low(), fq.input_high(),
                                        fq.output_low(), fq.output_high(), fq.levels(), fq.auto_broadcast())
            ->getResult(0);
}

//
// ConvertNearestToStridedConcatPass
//

class ConvertNearestToStridedConcatPass final :
        public IE::ConvertNearestToStridedConcatBase<ConvertNearestToStridedConcatPass> {
public:
    explicit ConvertNearestToStridedConcatPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class NearestInterpolateOpConverter;

private:
    void safeRunOnFunc() final;
};

class ConvertNearestToStridedConcatPass::NearestInterpolateOpConverter final :
        public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    NearestInterpolateOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::InterpolateOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertNearestToStridedConcatPass::NearestInterpolateOpConverter::matchAndRewrite(
        IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto inputShape = getShape(origOp.input());
    const auto outShape = parseIntArrayAttr<int64_t>(origOp.sizes_attrAttr());
    VPUX_THROW_UNLESS(inputShape.size() == 4, "Input shape must be 4d");
    VPUX_THROW_UNLESS(outShape.size() == 2, "Wrong number of spatial dims");
    // TODO: if out shape is not N times of in shape
    VPUX_THROW_UNLESS(outShape[1] % inputShape[Dims4D::Act::W] == 0 && outShape[0] % inputShape[Dims4D::Act::H] == 0,
                      "Only support N times upsampling");
    const auto scaleX = outShape[1] / inputShape[Dims4D::Act::W];
    const auto scaleY = outShape[0] / inputShape[Dims4D::Act::H];

    const auto inputFQ = origOp.input().getDefiningOp<IE::FakeQuantizeOp>();
    const auto outputFQ = !(origOp->getResult(0).use_empty())
                                  ? mlir::dyn_cast<IE::FakeQuantizeOp>(*(origOp->getResult(0).user_begin()))
                                  : nullptr;

    SmallVector<mlir::Value> widthSlices;
    SmallVector<mlir::Value> heightSlices;
    // Here is an assumption : scaleX !=0 AND scaleY !=0 as output shape is non-zero
    for (int i = 0; i < scaleY; ++i) {
        for (int j = 0; j < scaleX; ++j) {
            widthSlices.push_back(origOp.input());
        }

        auto newOp = widthSlices.size() != 1
                             ? rewriter.create<IE::ConcatOp>(origOp->getLoc(), widthSlices, Dims4D::Act::W, 1, scaleX)
                                       .output()
                             : widthSlices.front();

        // TODO remove this propagation after moving such functionality to Q-D propagation pass
        if (inputFQ != nullptr && outputFQ != nullptr && widthSlices.size() != 0) {
            newOp = createFQ(rewriter, newOp, outputFQ);
        }

        heightSlices.push_back(newOp);
        widthSlices.clear();
    }
    const auto resultConcat = heightSlices.size() != 1 ? rewriter.create<IE::ConcatOp>(origOp->getLoc(), heightSlices,
                                                                                       Dims4D::Act::H, 1, scaleY)
                                                       : heightSlices.front();
    rewriter.replaceOp(origOp, resultConcat);

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertNearestToStridedConcatPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::InterpolateOp>([&](IE::InterpolateOp op) {
        const auto attrs = op.attr();
        return !(attrs.mode().getValue() == IE::InterpolateMode::NEAREST && !attrs.antialias().getValue() &&
                 attrs.coord_mode().getValue() == IE::InterpolateCoordMode::ASYMMETRIC &&
                 attrs.shape_calc_mode().getValue() == IE::InterpolateCalcMode::SIZES &&
                 op.axes_attrAttr().size() == 2);
    });
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::FakeQuantizeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<NearestInterpolateOpConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertNearestToStridedConcatPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertNearestToStridedConcatPass(Logger log) {
    return std::make_unique<ConvertNearestToStridedConcatPass>(log);
}
