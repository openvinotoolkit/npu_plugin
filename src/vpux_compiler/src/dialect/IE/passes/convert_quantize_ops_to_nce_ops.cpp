//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/passes.hpp"

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
    class DequantizeToDwRewriter;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

//
// QuantizeDequantizeToAvgPool
//

template <class ConcreteOp>
class QuantizeDequantizeToAvgPool final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    QuantizeDequantizeToAvgPool(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx, benefitLow), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult QuantizeDequantizeToAvgPool<ConcreteOp>::matchAndRewrite(ConcreteOp originOp,
                                                                             mlir::PatternRewriter& rewriter) const {
    auto newPooling = IE::createIdentityAvgPool(originOp.getInput(), originOp.getType(), rewriter, originOp->getLoc());
    rewriter.replaceOp(originOp, newPooling->getResult(0));
    return mlir::success();
}

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
    rewriter.replaceOpWithNewOp<IE::AndOp>(originOp, originOp.getType(), originOp.getInput(), originOp.getInput(),
                                           broadcastType, nullptr, nullptr);

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

    auto inElemType = originOp.getInput().getType().cast<vpux::NDTypeInterface>().getElementType();
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

    auto quantizeCastOp =
            rewriter.create<IE::QuantizeCastOp>(originOp.getLoc(), originOp.getInput(), outQuantizeElemType);

    rewriter.replaceOpWithNewOp<IE::AddOp>(originOp, originOp.getType(), quantizeCastOp.getResult(),
                                           quantizeCastOp.getResult(), broadcastType, nullptr, nullptr);

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

    auto outElemType = originOp.getOutput().getType().cast<vpux::NDTypeInterface>().getElementType();
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

    auto addOp = rewriter.create<IE::AddOp>(originOp.getLoc(), newAddOutType, originOp.getInput(), originOp.getInput(),
                                            broadcastType, nullptr, nullptr);

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
    if (auto quantizeType = elementType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        const std::vector<ov::float16> vals(OC, 1.f);
        const auto baseType = mlir::RankedTensorType::get({OC, 1, 1, 1}, mlir::Float16Type::get(ctx));
        const auto baseAttr = mlir::DenseElementsAttr::get(baseType, ArrayRef(vals));
        const auto contentAttr = Const::ContentAttr::get(baseAttr);
        const auto quantWeightsConstAttr =
                contentAttr.convertElemType(normalizeQuantStorageType(quantizeType)).quantCast(quantizeType);
        const auto weightsType = contentAttr.getType().cast<vpux::NDTypeInterface>().changeElemType(quantizeType);
        return rewriter.create<Const::DeclareOp>(loc, weightsType, quantWeightsConstAttr);
    } else {
        if (elementType.isF16()) {
            const std::vector<ov::float16> vals(OC, 1.f);
            const auto baseType = mlir::RankedTensorType::get({OC, 1, 1, 1}, mlir::Float16Type::get(ctx));
            const auto baseAttr = mlir::DenseElementsAttr::get(baseType, ArrayRef(vals));
            return rewriter.create<Const::DeclareOp>(loc, baseType, Const::ContentAttr::get(baseAttr));
        } else if (elementType.isF32()) {
            const std::vector<float> vals(OC, 1.f);
            const auto baseType = mlir::RankedTensorType::get({OC, 1, 1, 1}, mlir::Float32Type::get(ctx));
            const auto baseAttr = mlir::DenseElementsAttr::get(baseType, ArrayRef(vals));
            return rewriter.create<Const::DeclareOp>(loc, baseType, Const::ContentAttr::get(baseAttr));
        } else {
            VPUX_THROW("buildDwWeights: other types are not supported");
        }
    }
}

mlir::LogicalResult ConvertQuantizeOpsToNceOpsPass::QuantizeToDwRewriter::matchAndRewrite(
        IE::QuantizeOp originOp, mlir::PatternRewriter& rewriter) const {
    const auto origType = originOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto origShape = origType.getShape();
    const auto OC = origShape[Dims4D::Act::C];
    auto weights = buildDwWeights(originOp->getLoc(), OC, origType.getElementType(), rewriter);

    const auto ctx = rewriter.getContext();
    const auto attrStrides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    const auto attrPadsBegin = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto attrPadsEnd = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto dilationsAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});

    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(
            originOp, originOp.getOutput().getType(), originOp.getInput(), weights,
            /*bias=*/nullptr, attrStrides, attrPadsBegin, attrPadsEnd, dilationsAttr, getIntAttr(ctx, OC),
            /*post_opAttr=*/nullptr, /*clampAttr*/ nullptr);

    return mlir::success();
}

class ConvertQuantizeOpsToNceOpsPass::DequantizeToDwRewriter final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    DequantizeToDwRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp DequantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertQuantizeOpsToNceOpsPass::DequantizeToDwRewriter::matchAndRewrite(
        IE::DequantizeOp originOp, mlir::PatternRewriter& rewriter) const {
    const auto origType = originOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto origShape = origType.getShape();
    const auto OC = origShape[Dims4D::Act::C];
    const auto ctx = rewriter.getContext();

    const auto quantizeType = mlir::quant::UniformQuantizedType::get(
            /*flags=*/0, /*storageType=*/getUInt8Type(ctx), /*expressedType=*/mlir::Float16Type::get(ctx),
            /*scale=*/1.0, /*zeroPoint=*/0, /*storageTypeMin=*/0, /*storageTypeMax=*/255);
    auto quantWeightsOp = buildDwWeights(originOp->getLoc(), OC, quantizeType, rewriter);

    const auto attrStrides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    const auto attrPadsBegin = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto attrPadsEnd = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto dilationsAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});

    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(
            originOp, originOp.getOutput().getType(), originOp.getInput(), quantWeightsOp,
            /*bias=*/nullptr, attrStrides, attrPadsBegin, attrPadsEnd, dilationsAttr, getIntAttr(ctx, OC),
            /*post_opAttr=*/nullptr, /*clampAttr*/ nullptr);

    return mlir::success();
}

void ConvertQuantizeOpsToNceOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    const std::set<VPU::ArchKind> compatibleTargets = {
            VPU::ArchKind::VPUX37XX,
    };

    // HW Eltwise and AvgPool supports only per-tensor bias/scale parameters
    auto isLegalQuantizeOp = [&](IE::QuantizeOp quantizeOp) {
        auto outType = quantizeOp.getOutput().getType().cast<vpux::NDTypeInterface>();
        const auto isPerChannelQuantized = outType.getElementType().isa<mlir::quant::UniformQuantizedPerAxisType>();
        const auto canUseCMajor = VPU::NCEInvariant::isChannelMajorCompatible(arch, outType);

        auto outputLayerUsers = quantizeOp.getOutput().getUsers();
        auto anyUserIsConv = !outputLayerUsers.empty() && ::llvm::any_of(outputLayerUsers, [](auto user) {
            return mlir::isa<IE::ConvolutionOp>(user);
        });

        return (anyUserIsConv && canUseCMajor) || isPerChannelQuantized;
    };

    auto isLegalDequantizeOp = [](IE::DequantizeOp dequantizeOp) {
        auto inElemType = dequantizeOp.getInput().getType().cast<vpux::NDTypeInterface>().getElementType();
        return inElemType.isa<mlir::quant::UniformQuantizedPerAxisType>();
    };

    // perTensor quantize/dequantize convert to avgpool
    // E#98802 avgpool is faster than add for big input size.
    // avgpool support rank >= 3, and currently convert shape to 4D does not support avgpool, so here limit to rank = 4

    mlir::ConversionTarget toAvgPoolTarget(ctx);
    mlir::RewritePatternSet toAvgPoolPatterns(&ctx);
    if (compatibleTargets.count(arch) > 0) {
        toAvgPoolTarget.addDynamicallyLegalOp<IE::QuantizeOp>([&](IE::QuantizeOp quantizeOp) {
            auto inType = quantizeOp.getInput().getType().cast<vpux::NDTypeInterface>();
            auto inRank = inType.getRank();
            return isLegalQuantizeOp(quantizeOp) || inRank != 4 ||
                   inType.getTotalAllocSize() <= vpux::VPU::getTotalCMXSize(quantizeOp);
        });
        toAvgPoolTarget.addDynamicallyLegalOp<IE::DequantizeOp>([&](IE::DequantizeOp dequantizeOp) {
            auto inType = dequantizeOp.getInput().getType().cast<vpux::NDTypeInterface>();
            auto inRank = inType.getRank();
            return isLegalDequantizeOp(dequantizeOp) || inRank != 4 ||
                   inType.getTotalAllocSize() <= vpux::VPU::getTotalCMXSize(dequantizeOp);
        });
        toAvgPoolTarget.addLegalOp<IE::AvgPoolOp>();

        toAvgPoolPatterns.add<QuantizeDequantizeToAvgPool<IE::QuantizeOp>>(&ctx, _log);
        toAvgPoolPatterns.add<QuantizeDequantizeToAvgPool<IE::DequantizeOp>>(&ctx, _log);
    }
    if (mlir::failed(mlir::applyPartialConversion(func, toAvgPoolTarget, std::move(toAvgPoolPatterns)))) {
        signalPassFailure();
    }

    // perTensor quantize/dequantize convert to add or and
    mlir::ConversionTarget toEltwiseTarget(ctx);
    toEltwiseTarget.addDynamicallyLegalOp<IE::QuantizeOp>([&](IE::QuantizeOp quantizeOp) {
        return isLegalQuantizeOp(quantizeOp);
    });
    toEltwiseTarget.addDynamicallyLegalOp<IE::DequantizeOp>([&](IE::DequantizeOp dequantizeOp) {
        return isLegalDequantizeOp(dequantizeOp);
    });
    toEltwiseTarget.addLegalOp<IE::AndOp>();
    toEltwiseTarget.addLegalOp<IE::AddOp>();
    toEltwiseTarget.addLegalOp<IE::QuantizeCastOp>();

    mlir::RewritePatternSet toEltwisePatterns(&ctx);
    if (compatibleTargets.count(arch) > 0) {
        toEltwisePatterns.add<DequantizeToAddRewriter>(&ctx, _log);
        toEltwisePatterns.add<QuantizeToAddRewriter>(&ctx, _log);
    } else {
        toEltwisePatterns.add<GenericConverter<IE::QuantizeOp>>(&ctx, _log);
        toEltwisePatterns.add<GenericConverter<IE::DequantizeOp>>(&ctx, _log);
    }

    if (mlir::failed(mlir::applyPartialConversion(func, toEltwiseTarget, std::move(toEltwisePatterns)))) {
        signalPassFailure();
    }

    // per-axis scales and per-tensor zero points quantize/dequantize convert to DW conv
    mlir::ConversionTarget quantToDwTarget(ctx);
    mlir::RewritePatternSet quantToDwPatterns(&ctx);
    if (compatibleTargets.count(arch) > 0) {
        quantToDwTarget.addDynamicallyLegalOp<IE::QuantizeOp>([&](IE::QuantizeOp quantizeOp) {
            auto outType = quantizeOp.getOutput().getType().cast<vpux::NDTypeInterface>();
            const auto isPerChannelQuantized = outType.getElementType().isa<mlir::quant::UniformQuantizedPerAxisType>();
            const auto canUseCMajor = VPU::NCEInvariant::isChannelMajorCompatible(arch, outType);

            auto outputLayerUsers = quantizeOp.getOutput().getUsers();
            auto anyUserIsConv = !outputLayerUsers.empty() && ::llvm::any_of(outputLayerUsers, [](auto user) {
                return mlir::isa<IE::ConvolutionOp>(user);
            });

            return (anyUserIsConv && canUseCMajor) || !isPerChannelQuantized;
        });

        quantToDwTarget.addDynamicallyLegalOp<IE::DequantizeOp>([&](IE::DequantizeOp dequantizeOp) {
            auto inType = dequantizeOp.getInput().getType().cast<vpux::NDTypeInterface>();
            const auto isPerChannelQuantized = inType.getElementType().isa<mlir::quant::UniformQuantizedPerAxisType>();
            auto outElemmentType = dequantizeOp.getOutput().getType().cast<vpux::NDTypeInterface>().getElementType();
            return !isPerChannelQuantized || !outElemmentType.isF16();
        });

        quantToDwTarget.addLegalOp<Const::DeclareOp>();
        quantToDwTarget.addLegalOp<IE::GroupConvolutionOp>();

        quantToDwPatterns.add<QuantizeToDwRewriter>(&ctx, _log);
        quantToDwPatterns.add<DequantizeToDwRewriter>(&ctx, _log);
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
