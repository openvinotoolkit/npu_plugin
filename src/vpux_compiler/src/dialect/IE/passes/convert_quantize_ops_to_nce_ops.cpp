//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/Value.h>

using namespace vpux;

namespace {

//
// ConvertQuantizeOpsToNceOpsPass
//

class ConvertQuantizeOpsToNceOpsPass final : public IE::ConvertQuantizeOpsToNceOpsBase<ConvertQuantizeOpsToNceOpsPass> {
public:
    explicit ConvertQuantizeOpsToNceOpsPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    class DequantizeToAddRewriter;
    class QuantizeToAddRewriter;
    class QuantizeToDwRewriter;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

//
// GenericConverter
//

template <class ConcreteOp>
class GenericConverter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    GenericConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx, benefitLow), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult GenericConverter<ConcreteOp>::matchAndRewrite(ConcreteOp originOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    const auto broadcastType =
            vpux::IE::AutoBroadcastTypeAttr::get(this->getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT);
    rewriter.replaceOpWithNewOp<IE::AndOp>(originOp, originOp.getType(), originOp.input(), originOp.input(),
                                           broadcastType, nullptr);

    return mlir::success();
}

//
// DequantizeOpConverter
//

class ConvertQuantizeOpsToNceOpsPass::DequantizeToAddRewriter final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    DequantizeToAddRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertQuantizeOpsToNceOpsPass::DequantizeToAddRewriter::matchAndRewrite(
        IE::DequantizeOp originOp, mlir::PatternRewriter& rewriter) const {
    const auto broadcastType =
            vpux::IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT);

    auto inElemType = originOp.input().getType().cast<vpux::NDTypeInterface>().getElementType();
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

class ConvertQuantizeOpsToNceOpsPass::QuantizeToAddRewriter final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    QuantizeToAddRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertQuantizeOpsToNceOpsPass::QuantizeToAddRewriter::matchAndRewrite(
        IE::QuantizeOp originOp, mlir::PatternRewriter& rewriter) const {
    const auto broadcastType =
            vpux::IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT);

    auto outElemType = originOp.output().getType().cast<vpux::NDTypeInterface>().getElementType();
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

class ConvertQuantizeOpsToNceOpsPass::QuantizeToDwRewriter final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    QuantizeToDwRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::Value buildDwWeights(const mlir::Location& loc, const int64_t OC, const mlir::Type& elementType,
                           mlir::PatternRewriter& rewriter) {
    const auto ctx = rewriter.getContext();
    if (elementType.isF16()) {
        const std::vector<ngraph::float16> vals(OC, 1.f);
        const auto baseType = mlir::RankedTensorType::get({OC, 1, 1, 1}, mlir::Float16Type::get(ctx));
        const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));
        return rewriter.create<Const::DeclareOp>(loc, baseType, Const::ContentAttr::get(baseAttr));
    } else if (elementType.isF32()) {
        const std::vector<float> vals(OC, 1.f);
        const auto baseType = mlir::RankedTensorType::get({OC, 1, 1, 1}, mlir::Float32Type::get(ctx));
        const auto baseAttr = mlir::DenseElementsAttr::get(baseType, makeArrayRef(vals));
        return rewriter.create<Const::DeclareOp>(loc, baseType, Const::ContentAttr::get(baseAttr));
    } else {
        VPUX_THROW("buildDwWeights: only float16 and float32 types are supported");
    }
}

mlir::LogicalResult ConvertQuantizeOpsToNceOpsPass::QuantizeToDwRewriter::matchAndRewrite(
        IE::QuantizeOp originOp, mlir::PatternRewriter& rewriter) const {
    const auto origType = originOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto origShape = origType.getShape();
    const auto OC = origShape[Dims4D::Act::C];
    auto weights = buildDwWeights(originOp->getLoc(), OC, origType.getElementType(), rewriter);

    const auto ctx = rewriter.getContext();
    const auto attrStrides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    const auto attrPadsBegin = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto attrPadsEnd = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto dilationsAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});

    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(
            originOp, originOp.output().getType(), originOp.input(), weights,
            /*bias=*/nullptr, attrStrides, attrPadsBegin, attrPadsEnd, dilationsAttr, getIntAttr(ctx, OC),
            /*post_opAttr=*/nullptr);

    return mlir::success();
}

void ConvertQuantizeOpsToNceOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    // HW Eltwise supports only per-tensor bias/scale parameters
    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::QuantizeOp>([&](IE::QuantizeOp quantizeOp) {
        auto outType = quantizeOp.output().getType().cast<vpux::NDTypeInterface>();
        const auto isPerChannelQuantized = outType.getElementType().isa<mlir::quant::UniformQuantizedPerAxisType>();
        const auto canUseCMajor = VPU::NCEInvariant::isChannelMajorCompatible(arch, outType);

        auto outputLayerUsers = quantizeOp.output().getUsers();
        auto anyUserIsConv = !outputLayerUsers.empty() && ::llvm::any_of(outputLayerUsers, [](auto user) {
            return mlir::isa<IE::ConvolutionOp>(user);
        });

        return (anyUserIsConv && canUseCMajor) || isPerChannelQuantized;
    });
    target.addDynamicallyLegalOp<IE::DequantizeOp>([&](IE::DequantizeOp dequantizeOp) {
        auto inElemType = dequantizeOp.input().getType().cast<vpux::NDTypeInterface>().getElementType();
        return inElemType.isa<mlir::quant::UniformQuantizedPerAxisType>();
    });
    target.addLegalOp<IE::AndOp>();
    target.addLegalOp<IE::AddOp>();
    target.addLegalOp<IE::QuantizeCastOp>();

    mlir::RewritePatternSet patterns(&ctx);
    if (arch == VPU::ArchKind::VPUX37XX) {
        patterns.add<DequantizeToAddRewriter>(&ctx, _log);
        patterns.add<QuantizeToAddRewriter>(&ctx, _log);
    } else {
        patterns.add<GenericConverter<IE::QuantizeOp>>(&ctx, _log);
        patterns.add<GenericConverter<IE::DequantizeOp>>(&ctx, _log);
    }

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }

    mlir::ConversionTarget quantToDwTarget(ctx);
    mlir::RewritePatternSet quantToDwPatterns(&ctx);
    if (arch == VPU::ArchKind::VPUX37XX) {
        quantToDwTarget.addDynamicallyLegalOp<IE::QuantizeOp>([&](IE::QuantizeOp quantizeOp) {
            auto outType = quantizeOp.output().getType().cast<vpux::NDTypeInterface>();
            const auto isPerChannelQuantized = outType.getElementType().isa<mlir::quant::UniformQuantizedPerAxisType>();
            const auto canUseCMajor = VPU::NCEInvariant::isChannelMajorCompatible(arch, outType);

            auto outputLayerUsers = quantizeOp.output().getUsers();
            auto anyUserIsConv = !outputLayerUsers.empty() && ::llvm::any_of(outputLayerUsers, [](auto user) {
                return mlir::isa<IE::ConvolutionOp>(user);
            });

            return (anyUserIsConv && canUseCMajor) || !isPerChannelQuantized;
        });
        quantToDwTarget.addLegalOp<Const::DeclareOp>();
        quantToDwTarget.addLegalOp<IE::GroupConvolutionOp>();

        quantToDwPatterns.add<QuantizeToDwRewriter>(&ctx, _log);
    }

    if (mlir::failed(mlir::applyPartialConversion(func, quantToDwTarget, std::move(quantToDwPatterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertQuantizeOpsToNceOpsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertQuantizeOpsToNceOpsPass(Logger log) {
    return std::make_unique<ConvertQuantizeOpsToNceOpsPass>(log);
}
