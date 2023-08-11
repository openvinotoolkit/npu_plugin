//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

Const::DeclareOp createShapeConstForBroadCast(mlir::PatternRewriter& rewriter, mlir::MLIRContext* ctx,
                                              mlir::Location loc, ShapeRef shape) {
    auto intType = getSInt64Type(ctx);
    const auto shapeStorageType = mlir::RankedTensorType::get({static_cast<int64_t>(shape.size())}, intType);
    const auto shapeDenseAttr = mlir::DenseElementsAttr::get(shapeStorageType, shape.raw());
    auto newContentAttr = Const::ContentAttr::get(shapeDenseAttr).convertElemType(getSInt32Type(ctx));
    return rewriter.create<Const::DeclareOp>(loc, shapeStorageType, newContentAttr);
}

mlir::Value createFQ(mlir::PatternRewriter& rewriter, mlir::Value input, IE::FakeQuantizeOp fq) {
    const auto outputType = fq.output().getType().cast<vpux::NDTypeInterface>();
    const auto newOutputType = outputType.changeShape(getShape(input));
    return rewriter
            .create<IE::FakeQuantizeOp>(fq.getLoc(), newOutputType, input, fq.input_low(), fq.input_high(),
                                        fq.output_low(), fq.output_high(), fq.levels(), fq.auto_broadcast())
            ->getResult(0);
}

//
// ConvertNearestToBroadcastOrStridedConcatPass
//

class ConvertNearestToBroadcastOrStridedConcatPass final :
        public IE::ConvertNearestToStridedConcatBase<ConvertNearestToBroadcastOrStridedConcatPass> {
public:
    explicit ConvertNearestToBroadcastOrStridedConcatPass(const bool interpolateAsSEOp, Logger log)
            : _interpolateAsSEOp(interpolateAsSEOp) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

public:
    class NearestToBroadcastConverter;
    class NearestToStridedConcatConverter;

private:
    void safeRunOnFunc() final;

private:
    bool _interpolateAsSEOp;
};

mlir::LogicalResult ConvertNearestToBroadcastOrStridedConcatPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (interpolateAsSEOp.hasValue()) {
        _interpolateAsSEOp = interpolateAsSEOp.getValue();
    }

    return mlir::success();
}

// NearestToBroadcastConverter

class ConvertNearestToBroadcastOrStridedConcatPass::NearestToBroadcastConverter final :
        public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    NearestToBroadcastConverter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::InterpolateOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertNearestToBroadcastOrStridedConcatPass::NearestToBroadcastConverter::matchAndRewrite(
        IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto& ctx = origOp.getContext();
    const auto outShape = getShape(origOp.output());

    if (!IE::isBroadCastInterpolate(origOp)) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::BroadcastOp>(
            origOp, origOp.input(), createShapeConstForBroadCast(rewriter, ctx, origOp->getLoc(), outShape), nullptr,
            IE::BroadcastTypeAttr::get(ctx, IE::BroadcastType::NUMPY));

    return mlir::success();
}

// NearestToStridedConcatConverter

class ConvertNearestToBroadcastOrStridedConcatPass::NearestToStridedConcatConverter final :
        public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    NearestToStridedConcatConverter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::InterpolateOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertNearestToBroadcastOrStridedConcatPass::NearestToStridedConcatConverter::matchAndRewrite(
        IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto inputShape = getShape(origOp.input());
    const auto outShape = getShape(origOp.output());

    int64_t outputW = 0;
    int64_t outputH = 0;

    const auto outShapeSize = outShape.size();
    if (outShapeSize == 2) {
        outputW = outShape[Dim(1)];
        outputH = outShape[Dim(0)];
    } else if (outShapeSize == 4) {
        outputW = outShape[Dims4D::Act::W];
        outputH = outShape[Dims4D::Act::H];
    } else {
        VPUX_THROW("Wrong number of spatial dims: {0}", outShapeSize);
    }

    // TODO: add support for cases where output dimension is not divisible by input dimension
    VPUX_THROW_UNLESS(outputW % inputShape[Dims4D::Act::W] == 0 && outputH % inputShape[Dims4D::Act::H] == 0,
                      "Only N times upsampling is supported");

    const auto scaleX = outputW / inputShape[Dims4D::Act::W];
    const auto scaleY = outputH / inputShape[Dims4D::Act::H];

    const auto inputFQ = origOp.input().getDefiningOp<IE::FakeQuantizeOp>();
    const auto outputFQ = !(origOp->getResult(0).use_empty())
                                  ? mlir::dyn_cast<IE::FakeQuantizeOp>(*(origOp->getResult(0).user_begin()))
                                  : nullptr;

    SmallVector<mlir::Value> widthSlices;
    SmallVector<mlir::Value> heightSlices;
    mlir::Value widthConcatOp;
    // Here is an assumption : scaleX !=0 AND scaleY !=0 as output shape is non-zero

    for (int j = 0; j < scaleX; ++j) {
        widthSlices.push_back(origOp.input());
    }

    widthConcatOp =
            widthSlices.size() != 1
                    ? rewriter.create<IE::ConcatOp>(origOp->getLoc(), widthSlices, Dims4D::Act::W, 1, scaleX).output()
                    : widthSlices.front();

    // TODO remove this propagation after moving such functionality to Q-D propagation pass
    if (inputFQ != nullptr && outputFQ != nullptr && widthSlices.size() != 0) {
        widthConcatOp = createFQ(rewriter, widthConcatOp, outputFQ);
    }
    for (int i = 0; i < scaleY; ++i) {
        heightSlices.push_back(widthConcatOp);
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

void ConvertNearestToBroadcastOrStridedConcatPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto isLegalConvertToStrideConcat = [&](IE::InterpolateOp op) {
        const auto attrs = op.attr();
        const bool validAxesAttrSize = (op.axes_attrAttr().size() == 2 || op.axes_attrAttr().size() == 4);
        const auto inputShape = getShape(op.input());
        const auto outShape = getShape(op.output());

        return attrs.getMode().getValue() == IE::InterpolateMode::NEAREST && !attrs.getAntialias().getValue() &&
               attrs.getCoordMode().getValue() == IE::InterpolateCoordMode::ASYMMETRIC && validAxesAttrSize &&
               (outShape[Dims4D::Act::W] % inputShape[Dims4D::Act::W] == 0) &&
               (outShape[Dims4D::Act::H] % inputShape[Dims4D::Act::H] == 0);
    };

    const auto isLegalConvertToBroadCast = [&](IE::InterpolateOp op) {
        return IE::isBroadCastInterpolate(op);
    };

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::InterpolateOp>([&](IE::InterpolateOp op) {
        if (_interpolateAsSEOp) {
            if (VPU::NCEInterpolateOp::isSupported(op, logCb, /*checkLayout=*/false, /*checkChannelAlignment=*/false)) {
                return true;
            }
        }

        return !(isLegalConvertToStrideConcat(op) || isLegalConvertToBroadCast(op));
    });
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::BroadcastOp>();
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::FakeQuantizeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    SmallVector<mlir::PatternBenefit> benefitLevels = getBenefitLevels(2);
    patterns.add<NearestToBroadcastConverter>(&ctx, benefitLevels[0], _log);
    patterns.add<NearestToStridedConcatConverter>(&ctx, benefitLevels[1], _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertNearestToBroadCastOrStridedConcatPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertNearestToBroadCastOrStridedConcatPass(const bool interpolateAsSEOp,
                                                                                         Logger log) {
    return std::make_unique<ConvertNearestToBroadcastOrStridedConcatPass>(interpolateAsSEOp, log);
}
