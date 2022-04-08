//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

#include <numeric>
using namespace vpux;

namespace {

mlir::Value getZerosConst(mlir::PatternRewriter& rewriter, Shape constShape, IE::UpsamplingOp origOp) {
    const auto elemType = origOp.input().getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto dataStorageType = mlir::RankedTensorType::get(to_small_vector(constShape), elemType);

    mlir::DenseElementsAttr denseElementVal = wrapData(dataStorageType, 0.0f);
    VPUX_THROW_UNLESS(denseElementVal != nullptr,
                      "Upsampling has incompatible data type {0}, only float16 or float32 are supported", elemType);

    return rewriter.create<Const::DeclareOp>(origOp.getLoc(), dataStorageType, Const::ContentAttr::get(denseElementVal))
            .output();
}

mlir::Value getConcatResult(mlir::PatternRewriter& rewriter, vpux::Dim axis, int64_t factor, mlir::Value input,
                            Shape constShape, IE::UpsamplingOp origOp) {
    auto constZeros = getZerosConst(rewriter, constShape, origOp);

    SmallVector<mlir::Value> concatInputs;
    concatInputs.push_back(input);
    for (int i = 0; i < factor - 1; i++) {
        concatInputs.push_back(constZeros);
    }

    return rewriter.create<IE::ConcatOp>(origOp.getLoc(), mlir::ValueRange(concatInputs), axis, 1, factor).output();
}

//
// ConvertUpsamplingToStridedConcatPass
//

class ConvertUpsamplingToStridedConcatPass final :
        public IE::ConvertUpsamplingToStridedConcatBase<ConvertUpsamplingToStridedConcatPass> {
public:
    explicit ConvertUpsamplingToStridedConcatPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class UpsamplingOpConverter;

private:
    void safeRunOnFunc() final;
};

// UpsamplingOpConverter
class ConvertUpsamplingToStridedConcatPass::UpsamplingOpConverter final :
        public mlir::OpRewritePattern<IE::UpsamplingOp> {
public:
    UpsamplingOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::UpsamplingOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::UpsamplingOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertUpsamplingToStridedConcatPass::UpsamplingOpConverter::matchAndRewrite(
        IE::UpsamplingOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Upsampling Op {0}", origOp->getLoc());

    const auto inputShape = getShape(origOp.input());

    const auto upsamplingFactorVectorTmp = parseIntArrayAttr<int64_t>(origOp.upsampling_factor());
    auto padsL = parseIntArrayAttr<int64_t>(origOp.pad_l());
    auto padsR = parseIntArrayAttr<int64_t>(origOp.pad_r());
    SmallVector<int64_t> upsamplingFactorVector = {1, upsamplingFactorVectorTmp[2], upsamplingFactorVectorTmp[1],
                                                   upsamplingFactorVectorTmp[0]};
    SmallVector<int64_t> padingLAtt = {0, padsL[2], padsL[1], padsL[0]};
    SmallVector<int64_t> padingRAtt = {0, padsR[2], padsR[1], padsR[0]};
    bool convertNow = (upsamplingFactorVectorTmp[2] != 1);
    bool needSlicing = false;
    Shape sliceShape(inputShape.raw());

    // The DMA only handle this scenario
    //   outputShape = inShape*upsamplingFactor
    // Or it can't use a DMA to describe it.
    // We can get below formula based on the upsampingOP's defination
    //   outputShape =  inShape + (inShape - 1) * (upsamplingFactorVector - 1) + padLVector + padRVector
    // It can be simplified as below when padL = 0
    //    outputShape=inShape*upsamplingFactorVector - (upsamplingFactorVector - 1) + padRVector
    auto calculatePaddingEndFun = [&](size_t ch, int64_t factor) {
        auto delta = factor - 1;
        if (padingRAtt[ch] < delta) {
            // This upsamplingOP can't be converted to DMA, and run the stride concat logic.
            convertNow = true;
            needSlicing = true;
            sliceShape[Dim(ch)] = (sliceShape[Dim(ch)] * factor) - (delta - padingRAtt[ch]);
            padingRAtt[ch] = 0;
        } else {
            padingRAtt[ch] -= delta;
            sliceShape[Dim(ch)] = sliceShape[Dim(ch)] * factor;
        }
    };

    auto upsamplingResult = origOp.input();
    for (size_t i = 0; i < upsamplingFactorVector.size(); i++) {
        if (upsamplingFactorVector[i] > 1) {
            calculatePaddingEndFun(i, upsamplingFactorVector[i]);
        }
    }
    if (convertNow) {
        for (size_t i = 0; i < upsamplingFactorVector.size(); i++) {
            if (upsamplingFactorVector[i] > 1 && inputShape[Dim(i)] > 1) {
                upsamplingResult = getConcatResult(rewriter, Dim(i), upsamplingFactorVector[i], upsamplingResult,
                                                   getShape(upsamplingResult).raw(), origOp);
            }
        }
        if (needSlicing && (sliceShape != inputShape)) {
            const Shape sliceOffsets{0, 0, 0, 0};
            auto slice = rewriter.create<IE::SliceOp>(origOp->getLoc(), upsamplingResult,
                                                      getIntArrayAttr(rewriter, sliceOffsets.raw()),
                                                      getIntArrayAttr(rewriter, sliceShape.raw()));
            upsamplingResult = {slice->getResult(0)};
        }
    } else {
        SmallVector<int64_t> newPadL(3, 0);
        SmallVector<int64_t> newPadR = upsamplingFactorVectorTmp;
        for (size_t i = 0; i < newPadR.size(); i++) {
            newPadR[i] = newPadR[i] - 1;
        }
        auto newUpsample = rewriter.create<IE::UpsamplingOp>(
                origOp->getLoc(), upsamplingResult, origOp.upsampling_factor(), getIntArrayAttr(rewriter, newPadL),
                getIntArrayAttr(rewriter, newPadR));
        upsamplingResult = newUpsample.output();
    }
    auto isZero = [](auto val) {
        return val == 0;
    };
    if ((!llvm::all_of(padingLAtt, isZero)) || (!llvm::all_of(padingRAtt, isZero))) {
        auto padBeginAttr = getIntArrayAttr(rewriter, padingLAtt);
        auto padEndAttr = getIntArrayAttr(rewriter, padingRAtt);
        auto zeroFpAttr = getFPAttr(rewriter, 0.0f);
        auto padingOp = rewriter.create<IE::PadOp>(origOp->getLoc(), upsamplingResult, nullptr, nullptr, nullptr,
                                                   padBeginAttr, padEndAttr, zeroFpAttr, IE::PadMode::CONSTANT);
        upsamplingResult = padingOp.output();
    }

    rewriter.replaceOp(origOp, upsamplingResult);
    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertUpsamplingToStridedConcatPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::ConversionTarget target(ctx);

    auto module = getOperation();
    const auto arch = VPU::getArch(module);
    if (arch != VPU::ArchKind::VPUX37XX) {
        _log.trace("The conversion is only enabled for VPUX37XX device. Got architecture: {0}", arch);
        return;
    }

    target.addDynamicallyLegalOp<IE::UpsamplingOp>([&](IE::UpsamplingOp op) {
        const auto inputShape = getShape(op.input());
        const auto padLVector = parseIntArrayAttr<int64_t>(op.pad_l());
        const auto padRVector = parseIntArrayAttr<int64_t>(op.pad_r());
        const auto upsamplingFactorVector = parseIntArrayAttr<int64_t>(op.upsampling_factor());

        // Upsampling only supports 4D Input shape
        // Upsampling supports pads only for 3 axes
        // Upsampling supports factors only for 3 axes
        if (inputShape.size() != 4 || padLVector.size() != 3 || padRVector.size() != 3 ||
            upsamplingFactorVector.size() != 3) {
            return true;
        }

        // Based on upsamplingOP's defination, the input/output's shape has below relation
        //      outputShape=inShape*upsamplingFactorVector - (upsamplingFactorVector - 1) + padLVector + padRVector
        // To make the upsampingOP can be converted to DMA. It should promise
        //      outputShape = inShape*upsamplingFactor
        // So padL = 0 and padRVector = upsamplingFactorVector - 1
        int64_t initValue = 0;
        int64_t sumPadL = std::accumulate(padLVector.begin(), padLVector.end(), initValue);
        if (sumPadL) {
            return false;
        }

        for (size_t i = 0; i < upsamplingFactorVector.size(); i++) {
            auto delta = upsamplingFactorVector[i] - padRVector[i];
            if (1 != delta) {
                return false;
            }
        }
        return true;
    });

    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::PadOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<UpsamplingOpConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertUpsamplingToStridedConcatPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertUpsamplingToStridedConcatPass(Logger log) {
    return std::make_unique<ConvertUpsamplingToStridedConcatPass>(log);
}
