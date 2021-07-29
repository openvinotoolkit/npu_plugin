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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/IE/loop.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

// change storage type to U8 and shift zp, min, max attributes by the value of storage type min
mlir::quant::QuantizedType changeStorageTypeToU8(mlir::quant::QuantizedType originQType) {
    if (const auto uniformType = originQType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        const auto low = uniformType.getStorageTypeMin();

        return mlir::quant::UniformQuantizedType::get(
                0, getUInt8Type(uniformType.getContext()), uniformType.getExpressedType(), uniformType.getScale(),
                uniformType.getZeroPoint() - low, uniformType.getStorageTypeMin() - low,
                uniformType.getStorageTypeMax() - low);
    } else if (const auto perAxisType = originQType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto low = perAxisType.getStorageTypeMin();
        const auto zeroPoints = perAxisType.getZeroPoints();

        SmallVector<int64_t> newZeroPoints(zeroPoints.size());
        std::transform(zeroPoints.begin(), zeroPoints.end(), newZeroPoints.begin(), [low](int64_t zp) {
            return zp - low;
        });

        return mlir::quant::UniformQuantizedPerAxisType::get(
                0, getUInt8Type(perAxisType.getContext()), perAxisType.getExpressedType(), perAxisType.getScales(),
                newZeroPoints, perAxisType.getQuantizedDimension(), perAxisType.getStorageTypeMin() - low,
                perAxisType.getStorageTypeMax() - low);
    }

    VPUX_THROW("Unsupported Quantized Type '{0}'", originQType);
}

class ConstRewriter final : public mlir::OpRewritePattern<Const::DeclareOp> {
public:
    ConstRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<Const::DeclareOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(Const::DeclareOp constOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConstRewriter::matchAndRewrite(Const::DeclareOp constOp, mlir::PatternRewriter& rewriter) const {
    const auto constElemType = constOp.getType().cast<mlir::RankedTensorType>().getElementType();
    const auto quantType = constElemType.dyn_cast_or_null<mlir::quant::QuantizedType>();
    VPUX_THROW_UNLESS(quantType != nullptr, "Got non-quantized type {0}", constElemType);

    _log.trace("Convert weights '{0}' with quant type '{1}'", constOp, quantType);

    auto low = quantType.getStorageTypeMin();
    const auto qElemType = changeStorageTypeToU8(quantType);
    _log.trace("Use quantized element type '{0}'", qElemType);

    const auto qType = changeElemType(constOp.getType().cast<mlir::RankedTensorType>(), qElemType);
    const auto newInConstAttr = constOp.contentAttr()
                                        .quantCast()
                                        .convertElemType(getInt32Type(getContext()))
                                        .add(checked_cast<double>(-low))
                                        .convertElemType(getUInt8Type(getContext()))
                                        .quantCast(qElemType);

    rewriter.replaceOpWithNewOp<Const::DeclareOp>(constOp, qType, newInConstAttr);
    return mlir::success();
}

//
// ConvertWeightsToU8
//

class ConvertWeightsToU8Pass final : public IE::ConvertWeightsToU8Base<ConvertWeightsToU8Pass> {
public:
    explicit ConvertWeightsToU8Pass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertWeightsToU8Pass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<Const::DeclareOp>([&](Const::DeclareOp constOp) {
        const auto constElemType = constOp.getType().cast<mlir::RankedTensorType>().getElementType();
        const auto quantType = constElemType.dyn_cast_or_null<mlir::quant::QuantizedType>();
        return quantType == nullptr || !quantType.isSigned();
    });

    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.insert<ConstRewriter>(&ctx, _log);
    if (mlir::failed(applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createConvertWeightsToU8Pass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertWeightsToU8Pass(Logger log) {
    return std::make_unique<ConvertWeightsToU8Pass>(log);
}
