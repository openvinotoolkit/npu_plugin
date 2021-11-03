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

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/Value.h>

using namespace vpux;

namespace {

//
// ConvertQuantizeOpsToEltwisePass
//

class ConvertQuantizeOpsToEltwisePass final :
        public IE::ConvertQuantizeOpsToEltwiseBase<ConvertQuantizeOpsToEltwisePass> {
public:
    explicit ConvertQuantizeOpsToEltwisePass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    class DequantizeOpConverter;
    class QuantizeOpConverter;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

//
// DequantizeOpConverter
//

class ConvertQuantizeOpsToEltwisePass::DequantizeOpConverter final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    DequantizeOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertQuantizeOpsToEltwisePass::DequantizeOpConverter::matchAndRewrite(
        IE::DequantizeOp originOp, mlir::PatternRewriter& rewriter) const {
    const auto broadcastType =
            vpux::IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT);

    auto inElemType = originOp.input().getType().cast<mlir::ShapedType>().getElementType();
    auto uniformQInElemType = inElemType.dyn_cast<mlir::quant::UniformQuantizedType>();
    const auto scale = uniformQInElemType.getScale();
    // originQElemType = <u8:fp32, scale>
    // newQElemType = <u8:fp32, scale / 2>
    // Op -> originQElemType -> QuantizeCastOp -> newQElemType -> AddOp(output x2) -> result
    const auto newScale = static_cast<double>(scale / 2.0);
    const auto zeroPoint = uniformQInElemType.getZeroPoint();

    auto qType = inElemType.dyn_cast<mlir::quant::QuantizedType>();
    auto outQuantizeElemType = mlir::quant::UniformQuantizedType::get(
            qType.getFlags(), qType.getStorageType(), qType.getExpressedType(), newScale, zeroPoint,
            qType.getStorageTypeMin(), qType.getStorageTypeMax());

    auto quantizeCastOp = rewriter.create<IE::QuantizeCastOp>(originOp.getLoc(), originOp.input(), outQuantizeElemType);

    rewriter.replaceOpWithNewOp<IE::AddOp>(originOp, originOp.getType(), quantizeCastOp.getResult(),
                                           quantizeCastOp.getResult(), broadcastType, nullptr);

    return mlir::success();
}

//
// QuantizeOpConverter
//

class ConvertQuantizeOpsToEltwisePass::QuantizeOpConverter final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    QuantizeOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertQuantizeOpsToEltwisePass::QuantizeOpConverter::matchAndRewrite(
        IE::QuantizeOp originOp, mlir::PatternRewriter& rewriter) const {
    const auto broadcastType =
            vpux::IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT);

    auto outElemType = originOp.output().getType().cast<mlir::ShapedType>().getElementType();
    auto uniformQOutElemType = outElemType.dyn_cast<mlir::quant::UniformQuantizedType>();
    const auto scale = uniformQOutElemType.getScale();
    // originQElemType = <u8:fp32, scale>
    // newQElemType = <u8:fp32, scale * 2>
    // Op -> AddOp(output x2) -> newQElemType -> QuantizeCastOp -> originQElemType -> result
    const auto newScale = static_cast<double>(scale * 2.0);
    const auto zeroPoint = uniformQOutElemType.getZeroPoint();

    auto qType = outElemType.dyn_cast<mlir::quant::QuantizedType>();
    auto quantizeElemType = mlir::quant::UniformQuantizedType::get(
            qType.getFlags(), qType.getStorageType(), qType.getExpressedType(), newScale, zeroPoint,
            qType.getStorageTypeMin(), qType.getStorageTypeMax());
    auto newAddOutType = mlir::RankedTensorType::get(originOp.getType().getShape(), quantizeElemType);

    auto addOp = rewriter.create<IE::AddOp>(originOp.getLoc(), newAddOutType, originOp.input(), originOp.input(),
                                            broadcastType, nullptr);

    rewriter.replaceOpWithNewOp<IE::QuantizeCastOp>(originOp, addOp.getResult(), outElemType);

    return mlir::success();
}

void ConvertQuantizeOpsToEltwisePass::safeRunOnFunc() {
    auto& ctx = getContext();

    // HW Eltwise supports only per-tensor bias/scale parameters
    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::QuantizeOp>([&](IE::QuantizeOp quantizeOp) {
        auto outElemType = quantizeOp.output().getType().cast<mlir::ShapedType>().getElementType();
        return outElemType.isa<mlir::quant::UniformQuantizedPerAxisType>();
    });
    target.addDynamicallyLegalOp<IE::DequantizeOp>([&](IE::DequantizeOp dequantizeOp) {
        auto inElemType = dequantizeOp.input().getType().cast<mlir::ShapedType>().getElementType();
        return inElemType.isa<mlir::quant::UniformQuantizedPerAxisType>();
    });
    target.addLegalOp<IE::AddOp>();
    target.addLegalOp<IE::QuantizeCastOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<DequantizeOpConverter>(&ctx, _log);
    patterns.insert<QuantizeOpConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertQuantizeOpsToEltwisePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertQuantizeOpsToEltwisePass(Logger log) {
    return std::make_unique<ConvertQuantizeOpsToEltwisePass>(log);
}
