//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

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

// padding Right or bottom for given input
mlir::Value createPadding(mlir::PatternRewriter& rewriter, IE::InterpolateOp origOp, mlir::Value input, Dim axis,
                          int64_t scale) {
    auto inputShape = getShape(input);
    auto offsets = SmallVector<int64_t>(inputShape.size(), 0);
    auto sizes = SmallVector<int64_t>(inputShape.begin(), inputShape.end());
    offsets[axis.ind()] = inputShape[axis] - 1;
    sizes[axis.ind()] = 1;

    auto subSlice = rewriter.create<IE::SliceOp>(origOp->getLoc(), input, getIntArrayAttr(origOp.getContext(), offsets),
                                                 getIntArrayAttr(origOp.getContext(), sizes))
                            .result();

    SmallVector<mlir::Value> subSlices;
    subSlices.push_back(input);
    subSlices.insert(subSlices.end(), scale - 1, subSlice);
    return rewriter.create<IE::ConcatOp>(origOp->getLoc(), subSlices, axis).output();
}

mlir::Value createAverageDWConv(mlir::Value input, ShapeRef kernelShape, mlir::Location loc, IE::FakeQuantizeOp inputFQ,
                                mlir::PatternRewriter& rewriter, Logger log) {
    log.nest().trace("Create dw conv {0}: kernel {1}", loc, kernelShape);
    auto inShape = getShape(input);
    auto dilationsAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{1, 1});
    auto stridesAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{1, 1});
    auto padBeginAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{0, 0});
    auto padEndAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{0, 0});
    auto groupAttr = getIntAttr(rewriter, inShape[Dims4D::Act::C]);

    const auto elemType = input.getType().cast<vpux::NDTypeInterface>().getElementType();
    auto createConstOp = [&](ShapeRef shape, float16 value) -> Const::DeclareOp {
        const auto dataStorageType = mlir::RankedTensorType::get(to_small_vector(shape), elemType);

        const auto denseElementVal = mlir::DenseElementsAttr::get(dataStorageType, value);
        return rewriter.create<Const::DeclareOp>(loc, dataStorageType, Const::ContentAttr::get(denseElementVal));
    };

    // OC is equal with IC
    const auto weightShape = Shape{inShape[Dims4D::Act::C], 1, kernelShape[Dim(0)], kernelShape[Dim(1)]};
    const float weightsScaleFactor = 1.0f / static_cast<float>(kernelShape[Dim(0)] * kernelShape[Dim(1)]);
    const float weightRealVal = (inputFQ != nullptr) ? 1.0f : weightsScaleFactor;
    auto dwConvFilter = createConstOp(weightShape, weightRealVal);
    auto weights = dwConvFilter.getOutput();
    const auto dataStorageType = mlir::RankedTensorType::get(to_small_vector(weightShape), elemType);
    // Add fakeQuan after kernel if needed
    if (inputFQ != nullptr) {
        const auto fqArgType = mlir::RankedTensorType::get({}, elemType);

        auto fqLevelsVal = getIntAttr(rewriter, 255);
        auto fqLowVal = VPU::declareFloatConst(rewriter, loc, 0.0f, fqArgType);
        auto fqInHighVal = VPU::declareFloatConst(rewriter, loc, 254.0f, fqArgType);
        auto fqOutHighVal = VPU::declareFloatConst(rewriter, loc, 254.0f * weightsScaleFactor, fqArgType);

        auto quantizationForWeights =
                rewriter.create<IE::FakeQuantizeOp>(loc, dataStorageType, weights, fqLowVal, fqInHighVal, fqLowVal,
                                                    fqOutHighVal, fqLevelsVal, inputFQ.auto_broadcastAttr());
        weights = quantizationForWeights.output();
    }

    auto newLoc = appendLoc(loc, "_interpolate_GroupConv_{0}_{1}", kernelShape[Dim(0)], kernelShape[Dim(1)]);
    auto averageDWConv =
            rewriter.create<IE::GroupConvolutionOp>(newLoc, input, weights, /*bias=*/nullptr, stridesAttr, padBeginAttr,
                                                    padEndAttr, dilationsAttr, groupAttr, /*post_opAttr=*/nullptr);

    return averageDWConv.output();
}

mlir::Value createMaxPool(mlir::Value input, mlir::Location loc, mlir::PatternRewriter& rewriter) {
    const SmallVector<int64_t> maxPoolStrides = {1, 1};
    const SmallVector<int64_t> maxPoolKernels = {1, 1};
    const SmallVector<int64_t> pads = {0, 0};
    const auto padsAttr = getIntArrayAttr(rewriter, pads);
    auto ctx = rewriter.getContext();

    return rewriter
            .create<IE::MaxPoolOp>(loc, input, getIntArrayAttr(rewriter, maxPoolKernels),
                                   getIntArrayAttr(rewriter, maxPoolStrides), padsAttr, padsAttr,
                                   vpux::IE::RoundingTypeAttr::get(ctx, vpux::IE::RoundingType::FLOOR), nullptr)
            .output();
}

//
// ConvertBilinearToStridedConcatAndConvPass
//

class ConvertBilinearToStridedConcatAndConvPass final :
        public IE::ConvertBilinearToStridedConcatAndConvBase<ConvertBilinearToStridedConcatAndConvPass> {
public:
    explicit ConvertBilinearToStridedConcatAndConvPass(const bool interpolateAsSEOp, Logger log)
            : _interpolateAsSEOp(interpolateAsSEOp) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

public:
    class BilinearInterpolateOpConverter;
    class BilinearInterpolateOpConverterV2;

private:
    void safeRunOnFunc() final;

private:
    bool _interpolateAsSEOp;
};

mlir::LogicalResult ConvertBilinearToStridedConcatAndConvPass::initialize(mlir::MLIRContext* ctx) {
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

// BilinearInterpolateOpConverter
class ConvertBilinearToStridedConcatAndConvPass::BilinearInterpolateOpConverter final :
        public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    BilinearInterpolateOpConverter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::InterpolateOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// BilinearInterpolateOpConverterV2
class ConvertBilinearToStridedConcatAndConvPass::BilinearInterpolateOpConverterV2 final :
        public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    BilinearInterpolateOpConverterV2(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::InterpolateOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

/// @brief Replace NxN bilinear interpolate as a cascaded structure of one NxN nearest interpolate with padding and
/// one depthwise convolution with NXN kernel.
/// @details Nearest part is implemented by strided concats over width and height,
/// padding part is implemented by slice-concat. See ticket: E43217.
/// @reminder Current solution can be optimized further by SEP feature in future
mlir::LogicalResult ConvertBilinearToStridedConcatAndConvPass::BilinearInterpolateOpConverter::matchAndRewrite(
        IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Get bilinear Interpolate Op {0}", origOp);
    const auto inputShape = getShape(origOp.input());
    const auto attrs = origOp.attr();
    const auto outShape = getShape(origOp.output());

    if ((inputShape[Dims4D::Act::N] != outShape[Dims4D::Act::N]) ||
        (inputShape[Dims4D::Act::C] != outShape[Dims4D::Act::C])) {
        VPUX_THROW("Interpolate axes must be H or W.");
    }

    bool isAlignCorners = 0;

    if ((attrs.getCoordMode().getValue() == IE::InterpolateCoordMode::ALIGN_CORNERS) &&
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

    SmallVector<mlir::Value> heightSlices(scaleH, newOp);
    auto nearestInterpolateOut =
            heightSlices.size() != 1
                    ? rewriter.create<IE::ConcatOp>(origOp->getLoc(), heightSlices, Dims4D::Act::H, 1, scaleH)
                    : heightSlices.front();

    auto tensorPadded = nearestInterpolateOut;
    if (!isAlignCorners) {
        auto tensorPaddedWidth =
                (scaleW - 1 > 0) ? createPadding(rewriter, origOp, nearestInterpolateOut, Dims4D::Act::W, scaleW)
                                 : nearestInterpolateOut;
        tensorPadded = (scaleH - 1 > 0) ? createPadding(rewriter, origOp, tensorPaddedWidth, Dims4D::Act::H, scaleH)
                                        : tensorPaddedWidth;
    }

    // Create depthwise convolution
    auto dwConv = createAverageDWConv(tensorPadded, {scaleH, scaleW}, origOp.getLoc(), inputFQ, rewriter, _log);
    rewriter.replaceOp(origOp, dwConv);

    return mlir::success();
}

/// @brief Replace 2x2 bilinear interpolate as three depthwise convolutions and one maxpool or six depthwise
/// convolutions if use cmx-concat, and a cascaded structure of strided concat over W and H
/// @details It is a faster solutoin than the original one above. See ticket: E#49791
mlir::LogicalResult ConvertBilinearToStridedConcatAndConvPass::BilinearInterpolateOpConverterV2::matchAndRewrite(
        IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("DPU Interp solution V2: optimize bilinear Interpolate Op {0}", origOp);
    if (!origOp->hasOneUse()) {
        return mlir::failure();
    }

    auto input = origOp.input();
    const auto inputShape = getShape(input);
    const auto outShape = getShape(origOp.output());

    if ((inputShape[Dims4D::Act::N] != outShape[Dims4D::Act::N]) ||
        (inputShape[Dims4D::Act::C] != outShape[Dims4D::Act::C])) {
        VPUX_THROW("Interpolate axes must be H or W.");
    }

    const auto scaleW = outShape[Dims4D::Act::W] / inputShape[Dims4D::Act::W];
    const auto scaleH = outShape[Dims4D::Act::H] / inputShape[Dims4D::Act::H];
    const auto attrs = origOp.attr();
    // This is a fast dpu solution only for 2x2 upsampling
    if (scaleW != 2 || scaleH != 2 || (attrs.getCoordMode().getValue() == IE::InterpolateCoordMode::ALIGN_CORNERS)) {
        return mlir::failure();
    }

    // This solution introduces many dw convs (four) so that channel alignment will lead
    // extra computation overhead and strided dmas by slice ops than original solution
    const auto elemType = origOp.output().getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto alignment = VPU::NCEInvariant::getAlignment(elemType);
    if (inputShape[Dims4D::Act::C] % alignment != 0) {
        return mlir::failure();
    }

    const auto inputFQ = input.getDefiningOp<IE::FakeQuantizeOp>();
    const auto outputFQ = mlir::dyn_cast<IE::FakeQuantizeOp>(*(origOp.output().user_begin()));
    const mlir::Location location = origOp.getLoc();

    // Right slice and padding
    auto concatW = createPadding(rewriter, origOp, input, Dims4D::Act::W, scaleW);

    // Bottom slice and padding
    auto concatWH = createPadding(rewriter, origOp, concatW, Dims4D::Act::H, scaleH);

    // Left slice
    auto concatWHShape = getShape(concatWH);
    auto offsets = SmallVector<int64_t>(concatWHShape.size(), 0);
    auto sizes = SmallVector<int64_t>(concatWHShape.begin(), concatWHShape.end());
    if (sizes[Dims4D::Act::W.ind()] <= 1) {
        return mlir::failure();
    }
    sizes[Dims4D::Act::W.ind()] -= 1;
    mlir::Value leftSlice =
            rewriter.create<IE::SliceOp>(location, concatWH, getIntArrayAttr(origOp.getContext(), offsets),
                                         getIntArrayAttr(origOp.getContext(), sizes));

    auto getAverageDWConv = [&](mlir::Value input, ShapeRef kernelShape) {
        auto DWConv = createAverageDWConv(input, kernelShape, location, inputFQ, rewriter, _log);
        if (outputFQ != nullptr) {
            DWConv = createFQ(rewriter, DWConv, outputFQ);
        }
        return DWConv;
    };

    auto type = origOp.output().getType().cast<vpux::NDTypeInterface>();
    auto module = origOp->getParentOfType<mlir::ModuleOp>();

    // The interpolate was divided into 4 conv. for the first concat, output is 1/2 of original output
    auto canUseCmxConcat = [&]() -> bool {
        // if it can be convert to CMX concat, the performance can be improved, but if not,
        // the extra convolution will decrease the performance, so here add the size check with output. input,
        // weights, and weights table to make it more accurate
        const auto arch = VPU::getArch(origOp);
        const auto weightTable = vpux::VPU::NCEInvariant::getWeightsTableSize(inputShape[Dims4D::Act::C]);
        int64_t outSize = type.getTotalAllocSize().count() / scaleH;
        int64_t inSize = outSize / scaleW;
        int64_t weightsSize = inputShape[Dims4D::Act::C] * Byte(type.getElemTypeSize()).count();

        int64_t typeSizeFactor = 2;
        if (inputFQ != nullptr) {
            inSize = inSize / typeSizeFactor;
            weightsSize = weightsSize / typeSizeFactor;
        }
        if (outputFQ != nullptr) {
            outSize = outSize / typeSizeFactor;
        }

        SmallVector<Byte> buffers = {Byte(inSize), Byte(outSize), Byte(weightsSize), weightTable};
        return vpux::VPU::calculateAlignedBuffersMemoryRequirement(arch, buffers) <= vpux::VPU::getTotalCMXSize(module);
    };

    // MaxPool has a faster execution speed than DWConv, so if cmxconcat cannot be used,
    // MaxPool is used. When cmxconcat is available, cmxconcat requires that the DistributionAttrs
    // of the dpu tasks before and after the concat are compatible. So we insert DWConv here so that
    // all dpu tasks before and after concat are of the same type and size,which can ensure that they
    // are assigned the same attributes and that concat can be converted to cmxconcat.

    auto originCopy = canUseCmxConcat() ? getAverageDWConv(input, {1, 1}) : createMaxPool(input, location, rewriter);

    // Average over concatW, kernel_h = 1, kernel_w = 2
    auto averagePoolOverW = getAverageDWConv(concatW, {1, 2});

    // Average over leftSlice, kernel_h = 2, kernel_w = 1
    auto averagePoolOverH = getAverageDWConv(leftSlice, {2, 1});

    // Average over concatWH, kernel_h = 2, kernel_w = 2
    auto averagePoolOverWH = getAverageDWConv(concatWH, {2, 2});

    // Join over W
    SmallVector<mlir::Value> widthSlices{originCopy, averagePoolOverW};
    auto joinOverW0 = rewriter.create<IE::ConcatOp>(location, widthSlices, Dims4D::Act::W, 1, 2).output();

    widthSlices.clear();
    widthSlices = {averagePoolOverH, averagePoolOverWH};
    auto joinOverW1 = rewriter.create<IE::ConcatOp>(location, widthSlices, Dims4D::Act::W, 1, 2).output();

    if (canUseCmxConcat()) {
        joinOverW0 = getAverageDWConv(joinOverW0, {1, 1});
        joinOverW1 = getAverageDWConv(joinOverW1, {1, 1});
        if (outputFQ != nullptr) {
            joinOverW1 = createFQ(rewriter, joinOverW1, outputFQ);
            joinOverW0 = createFQ(rewriter, joinOverW0, outputFQ);
        }
    }

    // Join over H
    SmallVector<mlir::Value> heightSlices{joinOverW0, joinOverW1};
    auto joinOverH = rewriter.create<IE::ConcatOp>(location, heightSlices, Dims4D::Act::H, 1, 2).output();

    rewriter.replaceOp(origOp, joinOverH);

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertBilinearToStridedConcatAndConvPass::safeRunOnFunc() {
    auto& ctx = getContext();

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

        const auto attrs = op.attr();
        const auto interpMode = attrs.getMode().getValue();
        const auto antiAlias = attrs.getAntialias().getValue();
        const auto coordMode = attrs.getCoordMode().getValue();
        const auto inputShape = getShape(op.input());
        const auto outShape = getShape(op.output());
        int64_t scaleW = 1;
        int64_t scaleH = 1;

        if ((interpMode != IE::InterpolateMode::LINEAR_ONNX && interpMode != IE::InterpolateMode::LINEAR) ||
            antiAlias ||
            (coordMode != IE::InterpolateCoordMode::ALIGN_CORNERS &&
             coordMode != IE::InterpolateCoordMode::ASYMMETRIC)) {
            return true;
        }

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

        // E46240: only this kind of align_corners is accurate and is supported
        if ((attrs.getCoordMode().getValue() == IE::InterpolateCoordMode::ALIGN_CORNERS)) {
            if ((outShape[Dims4D::Act::W] - 1) % (inputShape[Dims4D::Act::W] - 1) == 0 &&
                (outShape[Dims4D::Act::H] - 1) % (inputShape[Dims4D::Act::H] - 1) == 0) {
                scaleW = (outShape[Dims4D::Act::W] - 1) / (inputShape[Dims4D::Act::W] - 1);
                scaleH = (outShape[Dims4D::Act::H] - 1) / (inputShape[Dims4D::Act::H] - 1);
            } else {
                return true;
            }
        } else {
            // Only supports N times upsampling in ASYMMETRIC mode
            if (outShape[Dims4D::Act::W] % inputShape[Dims4D::Act::W] == 0 &&
                outShape[Dims4D::Act::H] % inputShape[Dims4D::Act::H] == 0) {
                scaleW = outShape[Dims4D::Act::W] / inputShape[Dims4D::Act::W];
                scaleH = outShape[Dims4D::Act::H] / inputShape[Dims4D::Act::H];
            } else {
                return true;
            }
        }
        // Deeplab-v3 WA: UPA implementation may be better for some big bilinear interpolates
        // Current solution will produce some extra DMAs as we need do padding by slice-concat, which may cause
        // some performance loss especially for big interpolates. In future, SEP may help to solve this issue.
        // Details see ticket: E43217
        // The scaleW 4 and scaleH 4 is more efficient on VPUX37XX.
        // Details see ticket: E56905
        const auto arch = VPU::getArch(op);
        return (arch != VPU::ArchKind::VPUX37XX && scaleW == 4 && scaleH == 4) || (scaleW > 4 && scaleH > 4);
    });

    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::MaxPoolOp>();
    target.addLegalOp<IE::FakeQuantizeOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::GroupConvolutionOp>();

    mlir::RewritePatternSet patterns(&ctx);
    /// @warning The insert order for patterns can't decide the real execuation order
    /// So we use the explicit declaration (PatternBenefit) in their class constrution function
    /// to control them. Here pattern V2 executed first
    patterns.insert<BilinearInterpolateOpConverterV2>(&ctx, vpux::benefitMid, _log);
    patterns.insert<BilinearInterpolateOpConverter>(&ctx, vpux::benefitLow, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertBilinearToStridedConcatAndConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertBilinearToStridedConcatAndConvPass(const bool interpolateAsSEOp,
                                                                                      Logger log) {
    return std::make_unique<ConvertBilinearToStridedConcatAndConvPass>(interpolateAsSEOp, log);
}
