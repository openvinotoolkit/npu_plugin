//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
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

static mlir::DenseElementsAttr wrapData(const mlir::RankedTensorType dataStorageType, float data) {
    const auto elemType = dataStorageType.getElementType();
    if (elemType.isF32()) {
        return mlir::DenseElementsAttr::get(dataStorageType, data);
    } else if (elemType.isF16()) {
        const ngraph::float16 weightHalfVal = data;
        return mlir::DenseElementsAttr::get(dataStorageType, weightHalfVal);
    }
    return nullptr;
}

static inline Const::DeclareOp declareFloatConst(mlir::Location loc, float val, mlir::RankedTensorType argType,
                                                 mlir::PatternRewriter& rewriter) {
    const auto denseElementVal = wrapData(argType, val);
    // Must never fail, given the 'RankedTensorOf<[F16, F32]>:$input,' declaration.
    VPUX_THROW_UNLESS(denseElementVal != nullptr,
                      "Average pool has incompatible data type {0}, only float16 or float32 are supported",
                      argType.getElementType());

    return rewriter.create<Const::DeclareOp>(loc, argType, Const::ContentAttr::get(denseElementVal));
}

//
// ConvertBilinearToStridedConcatAndConvPass
//

class ConvertBilinearToStridedConcatAndConvPass final :
        public IE::ConvertBilinearToStridedConcatAndConvBase<ConvertBilinearToStridedConcatAndConvPass> {
public:
    explicit ConvertBilinearToStridedConcatAndConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class BilinearInterpolateOpConverter;

private:
    void safeRunOnFunc() final;
};

// BilinearInterpolateOpConverter
class ConvertBilinearToStridedConcatAndConvPass::BilinearInterpolateOpConverter final :
        public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    BilinearInterpolateOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::InterpolateOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isInterpolateQuantized(IE::InterpolateOp& origOp) const;

    Logger _log;
};

bool ConvertBilinearToStridedConcatAndConvPass::BilinearInterpolateOpConverter::isInterpolateQuantized(
        IE::InterpolateOp& origOp) const {
    const mlir::Operation* inputOp = origOp.input().getDefiningOp();
    if (inputOp == nullptr) {
        _log.trace("Interpolate's input is the region argument. Assuming it is not quantized.");
        return false;
    }
    const bool isInputLayerFQ = mlir::isa<IE::FakeQuantizeOp>(inputOp);
    const auto outputLayerUsers = origOp.output().getUsers();
    bool areAllUsersFQ = !outputLayerUsers.empty() && ::llvm::all_of(outputLayerUsers, [](auto user) {
        return ::mlir::isa<IE::FakeQuantizeOp>(user);
    });
    return isInputLayerFQ && areAllUsersFQ;
}

/// @brief Replace NxN bilinear interpolate as a cascaded structure of one NxN nearest interpolate with padding and
/// one depthwise convolution with NXN kernel.
/// @details Nearest part is implemented by strided concats over width and height,
/// padding part is implemented by slice-concat. See ticket: E#43217.
/// @reminder Current solution can be optimized further by SEP feature in future
mlir::LogicalResult ConvertBilinearToStridedConcatAndConvPass::BilinearInterpolateOpConverter::matchAndRewrite(
        IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Get bilinear Interpolate Op {0}", origOp);
    const auto inputShape = getShape(origOp.input());
    const auto attrs = origOp.attr();
    const auto outShape = getShape(origOp.output());
    const auto axesValue = parseIntArrayAttr<int64_t>(origOp.axes_attrAttr());
    VPUX_THROW_UNLESS(llvm::all_of(axesValue,
                                   [](auto axes) {
                                       return axes > 1;
                                   }),
                      "Axes must be H or W");
    bool isAlignCorners = 0;

    if ((attrs.coord_mode().getValue() == IE::InterpolateCoordMode::ALIGN_CORNERS) &&
        (outShape[Dims4D::Act::W] - 1) % (inputShape[Dims4D::Act::W] - 1) == 0 &&
        (outShape[Dims4D::Act::H] - 1) % (inputShape[Dims4D::Act::H] - 1) == 0) {
        isAlignCorners = 1;
    }

    auto scaleW = outShape[Dims4D::Act::W] / inputShape[Dims4D::Act::W];
    auto scaleH = outShape[Dims4D::Act::H] / inputShape[Dims4D::Act::H];

    if (isAlignCorners) {
        scaleW = (outShape[Dims4D::Act::W] - 1) / (inputShape[Dims4D::Act::W] - 1);
        scaleH = (outShape[Dims4D::Act::H] - 1) / (inputShape[Dims4D::Act::H] - 1);
    }

    const auto inputFQ = origOp.input().getDefiningOp<IE::FakeQuantizeOp>();

    SmallVector<mlir::Value> widthSlices(scaleW, origOp.input());
    auto newOp =
            widthSlices.size() != 1
                    ? rewriter.create<IE::ConcatOp>(origOp->getLoc(), widthSlices, Dims4D::Act::W, 1, scaleW).output()
                    : widthSlices.front();
    if (inputFQ != nullptr) {
        newOp = createFQ(rewriter, newOp, inputFQ);
    }

    SmallVector<mlir::Value> heightSlices(scaleH, newOp);
    auto nearestInterpolateOut =
            heightSlices.size() != 1
                    ? rewriter.create<IE::ConcatOp>(origOp->getLoc(), heightSlices, Dims4D::Act::H, 1, scaleH)
                    : heightSlices.front();
    if (inputFQ != nullptr) {
        nearestInterpolateOut = createFQ(rewriter, nearestInterpolateOut, inputFQ);
    }

    // padding Right and bottom
    auto createPadding = [&](mlir::Value input, Dim axes, int64_t scale) -> mlir::Value {
        auto inputShape = getShape(input);
        auto offsets = SmallVector<int64_t>(inputShape.size(), 0);
        auto sizes = SmallVector<int64_t>(inputShape.begin(), inputShape.end());
        offsets[axes.ind()] = outShape[axes] - 1;
        sizes[axes.ind()] = 1;

        const auto subSlice =
                rewriter.create<IE::SliceOp>(origOp->getLoc(), input, getIntArrayAttr(origOp.getContext(), offsets),
                                             getIntArrayAttr(origOp.getContext(), sizes));

        SmallVector<mlir::Value> subSlices;
        subSlices.push_back(input);
        subSlices.insert(subSlices.end(), scale - 1, subSlice->getResult(0));
        mlir::Value concatOp = rewriter.create<IE::ConcatOp>(origOp->getLoc(), subSlices, axes);

        if (inputFQ != nullptr) {
            concatOp = createFQ(rewriter, concatOp, inputFQ);
        }

        return concatOp;
    };

    auto tensorPadded = nearestInterpolateOut;
    if (!isAlignCorners) {
        auto tensorPaddedWidth =
                (scaleW - 1 > 0) ? createPadding(nearestInterpolateOut, Dims4D::Act::W, scaleW) : nearestInterpolateOut;
        tensorPadded = (scaleH - 1 > 0) ? createPadding(tensorPaddedWidth, Dims4D::Act::H, scaleH) : tensorPaddedWidth;
    }

    // Create depthwise Convolution
    auto dilationsAttr = getIntArrayAttr(origOp.getContext(), SmallVector<int32_t>{1, 1});
    auto stridesAttr = getIntArrayAttr(origOp.getContext(), SmallVector<int32_t>{1, 1});
    auto padBeginAttr = getIntArrayAttr(origOp.getContext(), SmallVector<int32_t>{0, 0});
    auto padEndAttr = getIntArrayAttr(origOp.getContext(), SmallVector<int32_t>{0, 0});
    auto groupAttr = getIntAttr(origOp.getContext(), outShape[Dims4D::Act::C]);

    const auto elemType = origOp.output().getType().cast<vpux::NDTypeInterface>().getElementType();
    auto createConstOp = [&](ShapeRef shape, float16 value) -> Const::DeclareOp {
        const auto dataStorageType = mlir::RankedTensorType::get(to_small_vector(shape), elemType);

        const auto denseElementVal = mlir::DenseElementsAttr::get(dataStorageType, value);
        return rewriter.create<Const::DeclareOp>(origOp.getLoc(), dataStorageType,
                                                 Const::ContentAttr::get(denseElementVal));
    };

    const auto quantizedEn = isInterpolateQuantized(origOp);
    const auto weightShape = Shape{outShape[Dims4D::Act::C], 1, scaleH, scaleW};
    const float weightsScaleFactor = 1.0f / static_cast<float>(scaleH * scaleW);
    const float weightRealVal = (quantizedEn) ? 1.0f : weightsScaleFactor;
    auto dwConvFilter = createConstOp(weightShape, weightRealVal);

    auto weights = dwConvFilter.output();

    const auto& ctx = origOp.getContext();
    const mlir::Location location = origOp.getLoc();
    const auto dataStorageType = mlir::RankedTensorType::get(to_small_vector(weightShape), elemType);
    if (quantizedEn) {
        const auto fqArgType = mlir::RankedTensorType::get({}, elemType);

        auto fqLevelsVal = getIntAttr(ctx, 255);
        auto fqLowVal = declareFloatConst(location, 0.0f, fqArgType, rewriter);
        auto fqInHighVal = declareFloatConst(location, 254.0f, fqArgType, rewriter);
        auto fqOutHighVal = declareFloatConst(location, 254.0f * weightsScaleFactor, fqArgType, rewriter);

        IE::FakeQuantizeOp inputLayerFQ = origOp.input().getDefiningOp<IE::FakeQuantizeOp>();
        IE::FakeQuantizeOp quantizationForWeights = rewriter.create<IE::FakeQuantizeOp>(
                origOp.getLoc(), dataStorageType, dwConvFilter.output(), fqLowVal, fqInHighVal, fqLowVal, fqOutHighVal,
                fqLevelsVal, inputLayerFQ.auto_broadcastAttr());
        weights = quantizationForWeights.output();
    }

    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(origOp, tensorPadded, weights, /*bias=*/nullptr, stridesAttr,
                                                        padBeginAttr, padEndAttr, dilationsAttr, groupAttr,
                                                        /*post_opAttr=*/nullptr);

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertBilinearToStridedConcatAndConvPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::InterpolateOp>([&](IE::InterpolateOp op) {
        const auto attrs = op.attr();
        const auto inputShape = getShape(op.input());
        const auto outShape = getShape(op.output());
        bool isAlignCorners = 0;
        // Only support 4D Input shape
        if (inputShape.size() != 4) {
            return true;
        }

        // Small-channel models WA: when the channel size is smaller than the channel alignment
        // The alignment causes worse performance than UPA interpolation
        const auto elemType = op.output().getType().cast<vpux::NDTypeInterface>().getElementType();
        const auto alignment = VPU::NCEInvariant::getAlignment(elemType);
        if (inputShape[Dims4D::Act::C] < alignment) {
            return true;
        }

        // Runtime already has a efficient implementation for this case
        // And also current solution for this case will produce lots of DMAs, which is not efficient
        if (inputShape[Dims4D::Act::H] == 1 && inputShape[Dims4D::Act::W] == 1) {
            return true;
        }
        if ((attrs.coord_mode().getValue() == IE::InterpolateCoordMode::ALIGN_CORNERS) &&
            (outShape[Dims4D::Act::W] - 1) % (inputShape[Dims4D::Act::W] - 1) == 0 &&
            (outShape[Dims4D::Act::H] - 1) % (inputShape[Dims4D::Act::H] - 1) == 0) {
            isAlignCorners = 1;
        }
        // Only supports N times upsampling
        if (!isAlignCorners) {
            if (outShape[Dims4D::Act::W] % inputShape[Dims4D::Act::W] != 0 &&
                outShape[Dims4D::Act::H] % inputShape[Dims4D::Act::H] != 0) {
                return true;
            }
        }
        // Deeplab-v3 WA: UPA implementation may be better for some big bilinear interpolates
        // Current solution will produce some extra DMAs as we need do padding by slice-concat, which may cause
        // some performance loss especially for big interpolates. In future, SEP may help to solve this issue.
        // Details see ticket: E#43217
        auto scaleW = outShape[Dims4D::Act::W] / inputShape[Dims4D::Act::W];
        auto scaleH = outShape[Dims4D::Act::H] / inputShape[Dims4D::Act::H];

        if (isAlignCorners) {
            scaleW = (outShape[Dims4D::Act::W] - 1) / (inputShape[Dims4D::Act::W] - 1);
            scaleH = (outShape[Dims4D::Act::H] - 1) / (inputShape[Dims4D::Act::H] - 1);
        }

        if (scaleW >= 4 && scaleH >= 4) {
            return true;
        }
        return !((attrs.mode().getValue() == IE::InterpolateMode::LINEAR_ONNX ||
                  attrs.mode().getValue() == IE::InterpolateMode::LINEAR) &&
                 !attrs.antialias().getValue());
    });
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::FakeQuantizeOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::GroupConvolutionOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<BilinearInterpolateOpConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertBilinearToStridedConcatAndConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertBilinearToStridedConcatAndConvPass(Logger log) {
    return std::make_unique<ConvertBilinearToStridedConcatAndConvPass>(log);
}
