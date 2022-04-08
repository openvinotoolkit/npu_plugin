//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/eltwise_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_utils.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

#include <limits>

using namespace vpux;

namespace {

Const::DeclareOp createActSparsityMap(mlir::PatternRewriter& rewriter, mlir::Type type) {
    auto dataType = type.cast<vpux::NDTypeInterface>();
    auto ctx = rewriter.getContext();
    if (auto sparseType = type.dyn_cast<VPU::SparseTensorType>()) {
        dataType = sparseType.getData().cast<vpux::NDTypeInterface>();
    }
    const auto sparsityMapType = dataType.changeElemType(mlir::IntegerType::get(ctx, 1, mlir::IntegerType::Signless))
                                         .cast<mlir::RankedTensorType>();

    const auto dataAttr = mlir::DenseElementsAttr::get(sparsityMapType, /*splatValue=*/true);
    const auto content = Const::ContentAttr::get(dataAttr);

    return rewriter.create<Const::DeclareOp>(mlir::UnknownLoc::get(ctx), sparsityMapType, content);
}

mlir::ArrayAttr getIdentityDimPermutation(mlir::MLIRContext* ctx) {
    SmallVector<SmallVector<int64_t>> permutation = {{0}, {1}, {2}, {3}};
    return getIntArrayOfArray(ctx, permutation);
}

std::tuple<mlir::Value, Shape> insertExpandToAlign(mlir::PatternRewriter& rewriter, mlir::Value input,
                                                   int64_t alignment) {
    const auto inputType = input.getType().cast<vpux::NDTypeInterface>();

    const auto inputShape = inputType.getShape();
    const auto dimC = Dims4D::Act::C;

    auto expandedShape = inputShape.toValues();
    expandedShape[dimC] = alignVal(inputShape[dimC], alignment);
    SmallVector<int64_t> padsBegin(inputShape.size(), 0);
    SmallVector<int64_t> padsEnd(inputShape.size(), 0);
    padsEnd[dimC.ind()] = expandedShape[dimC] - inputShape[dimC];

    const mlir::Type outputType = inputType.changeShape(expandedShape);
    auto ctx = rewriter.getContext();
    auto expandOp = rewriter.create<VPU::ExpandOp>(input.getLoc(), outputType, input, getIntArrayAttr(ctx, padsBegin),
                                                   getIntArrayAttr(ctx, padsEnd));
    return std::make_tuple(expandOp.output(), expandedShape);
}

std::tuple<mlir::Value, Shape> insertAlignmentReshape(mlir::PatternRewriter& rewriter, mlir::Value input,
                                                      int64_t alignment) {
    const auto inputType = input.getType().cast<vpux::NDTypeInterface>();

    const auto inputShape = inputType.getShape();
    const auto totalElements = inputShape.totalSize();
    // Trying to uniformely distribute elements
    const auto desiredAxisSize = checked_cast<int64_t>(std::cbrt(totalElements));
    auto numC = alignVal(desiredAxisSize, alignment);
    if (totalElements % numC != 0) {
        numC = alignment;
    }
    const auto spatialRemainder = totalElements / numC;
    // approximate square shape of spatial
    int64_t numH = checked_cast<int64_t>(std::floor(std::sqrt(spatialRemainder)));
    while (numH > 1 && spatialRemainder % numH != 0) {
        --numH;
    }
    const auto numW = spatialRemainder / numH;
    const Shape newShape{1, numC, numH, numW};
    VPUX_THROW_WHEN(newShape.totalSize() != totalElements,
                    "New shape '{0}' doesnt contain same number of elements as original '{1}'", newShape, inputShape);
    const auto ctx = input.getContext();
    auto reshapeOp =
            rewriter.create<VPU::AffineReshapeOp>(input.getLoc(), inputType.changeShape(newShape), input,
                                                  getIdentityDimPermutation(ctx), getIntArrayAttr(ctx, newShape));
    return std::make_tuple(reshapeOp.output(), newShape);
}

void rewriteSparsityOpWithEltwiseOp(mlir::PatternRewriter& rewriter, mlir::Operation* originalOp, vpux::LogCb logCb) {
    const auto loc = originalOp->getLoc();
    auto ctx = originalOp->getContext();

    mlir::Value input = originalOp->getOperand(0);
    mlir::Value output = originalOp->getResult(0);
    vpux::NDTypeInterface targetEltwiseOutputType = output.getType();

    const auto arch = VPU::getArch(originalOp);
    const auto opType = VPU::EltwiseType::ADD;
    const bool allowDifferentScales = true;
    const bool allowDifferentZp = true;

    auto inputType = input.getType();
    auto alignment = VPU::NCEEltwiseOp::getInputChannelAlignmentImpl(inputType);
    // 37XX support compressed inputs(IC=4), while Eltwise require alignment of 16.
    // Eltwise result dont depend on shape, so wrap Eltwise by reshapes to suitable shape
    bool needAlignment = !vpux::VPU::NCEInvariant::isAligned(inputType, alignment, logCb);
    bool isAlignmentResolvedByReshape = false;
    const auto originalShape = targetEltwiseOutputType.getShape();
    Shape alignedShape;
    if (needAlignment) {
        if (originalShape.totalSize() % alignment == 0) {
            std::tie(input, alignedShape) = insertAlignmentReshape(rewriter, input, alignment);
            isAlignmentResolvedByReshape = true;
        } else {
            std::tie(input, alignedShape) = insertExpandToAlign(rewriter, input, alignment);
        }
        inputType = input.getType();
        targetEltwiseOutputType = targetEltwiseOutputType.changeShape(alignedShape);
        output = input;  // Used only by isNCEEltwiseSupported, where only shape and elementType checked
        // so can be used regardless sparsity
    }

    if (!VPU::isNCEEltwiseSupported(arch, mlir::ValueRange{input, input}, output, allowDifferentScales,
                                    allowDifferentZp, logCb)) {
        VPUX_THROW("Cannot lower [De]Sparsify to Eltwise because of HW requirements {0}", originalOp->getLoc());
    }

    vpux::NDTypeInterface outputTypeForPPEAttr = targetEltwiseOutputType;

    // To keep values half of scale is needed
    auto elementType = outputTypeForPPEAttr.getElementType();
    // dummy filled type with half of scale for fp16 case
    auto newType = mlir::quant::UniformQuantizedType::get(0, getUInt8Type(ctx), rewriter.getF16Type(),
                                                          /*scale=*/2.0,
                                                          /*zeroPoint=*/0,
                                                          /*storageTypeMin=*/0,
                                                          /*storageTypeMax=*/std::numeric_limits<uint8_t>::max());
    if (auto qOutputType = elementType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        const auto newScale = qOutputType.getScale() * 2.;
        // For real quantType we should use real values, except scale
        newType = mlir::quant::UniformQuantizedType::get(0, getUInt8Type(ctx), rewriter.getF16Type(), newScale,
                                                         qOutputType.getZeroPoint(), qOutputType.getStorageTypeMin(),
                                                         qOutputType.getStorageTypeMax());
    } else if (auto qOutputType = elementType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        VPUX_THROW("UniformQuantizedPerAxisType is not supported by NCEEltwiseOps");
    }

    outputTypeForPPEAttr = outputTypeForPPEAttr.changeElemType(newType);

    auto ppeTaskAttr =
            VPU::getNCEEltwisePPETaskAttr(inputType, inputType, outputTypeForPPEAttr, nullptr, loc, opType, ctx, arch);

    auto eltwiseOp = rewriter.create<VPU::NCEEltwiseOp>(originalOp->getLoc(), targetEltwiseOutputType, input, input,
                                                        VPU::EltwiseTypeAttr::get(ctx, opType), ppeTaskAttr,
                                                        /*multi_cluster_strategyAttr=*/nullptr,
                                                        /*is_inplace*/ nullptr);

    auto originalOutput = originalOp->getResult(0);
    if (needAlignment) {
        const vpux::NDTypeInterface originalOutputType = originalOutput.getType();
        if (isAlignmentResolvedByReshape) {
            rewriter.replaceOpWithNewOp<VPU::AffineReshapeOp>(originalOp, originalOutputType, eltwiseOp.output(),
                                                              getIdentityDimPermutation(ctx),
                                                              getIntArrayAttr(ctx, originalOutputType.getShape()));
        } else {
            SmallVector<int64_t> offsets(alignedShape.size(), 0);
            SmallVector<int64_t> sizes(originalShape.begin(), originalShape.end());
            rewriter.replaceOpWithNewOp<VPU::SliceOp>(originalOp, originalOutputType, eltwiseOp.output(),
                                                      getIntArrayAttr(ctx, offsets), getIntArrayAttr(ctx, sizes));
        }
    } else {
        originalOutput.replaceAllUsesWith(eltwiseOp.output());
        rewriter.eraseOp(originalOp);
    }
}

//
// LowerSparsityOpsPass
//

class LowerSparsityOpsPass final : public VPU::LowerSparsityOpsBase<LowerSparsityOpsPass> {
public:
    explicit LowerSparsityOpsPass(Optional<bool> maybeFakeSparsify, Logger log): _maybeFakeSparsify(maybeFakeSparsify) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    Optional<bool> _maybeFakeSparsify;
};

mlir::LogicalResult LowerSparsityOpsPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (!fakeSparsify.hasValue()) {
        return mlir::success();
    }
    if (_maybeFakeSparsify.hasValue()) {
        _log.trace("Overloading C++  createLowerSparsityOpsPass argument by MLIR variable");
    }
    _maybeFakeSparsify = fakeSparsify;
    return mlir::success();
}

//
// RewriteDesparsify
//

class RewriteDesparsify final : public mlir::OpRewritePattern<VPU::DesparsifyOp> {
public:
    RewriteDesparsify(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::DesparsifyOp>(ctx), _log(log) {
        setDebugName("RewriteDesparsify");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::DesparsifyOp desparsifyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult RewriteDesparsify::matchAndRewrite(VPU::DesparsifyOp desparsifyOp,
                                                       mlir::PatternRewriter& rewriter) const {
    const auto logCb = [&](const formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, desparsifyOp, "[{0}] {1}", this->getDebugName(), msg.str());
    };
    _log.trace("Got '{0}' at '{1}'", desparsifyOp->getName(), desparsifyOp->getLoc());

    const auto outputType = desparsifyOp.output().getType().cast<vpux::NDTypeInterface>();
    VPUX_THROW_WHEN(outputType.getElementType().isa<mlir::quant::UniformQuantizedPerAxisType>(),
                    "Could not convert Desparsify at '{0}' because the data is quantized per-axis",
                    desparsifyOp->getLoc());

    rewriteSparsityOpWithEltwiseOp(rewriter, desparsifyOp.getOperation(), logCb);
    return mlir::success();
}

//
// RewriteSparsify
//

class RewriteSparsify final : public mlir::OpRewritePattern<VPU::SparsifyOp> {
public:
    RewriteSparsify(mlir::MLIRContext* ctx, bool useFakeSparsify, Logger log)
            : mlir::OpRewritePattern<VPU::SparsifyOp>(ctx), _useFakeSparsify(useFakeSparsify), _log(log) {
        setDebugName("RewriteSparsify");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::SparsifyOp SparsifyOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _useFakeSparsify;
    Logger _log;
};

mlir::LogicalResult RewriteSparsify::matchAndRewrite(VPU::SparsifyOp sparsifyOp,
                                                     mlir::PatternRewriter& rewriter) const {
    const auto logCb = [&](const formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, sparsifyOp, "[{0}] {1}", this->getDebugName(), msg.str());
    };
    _log.trace("Got '{0}' at '{1}'", sparsifyOp->getName(), sparsifyOp->getLoc());

    const auto outputType = sparsifyOp.output().getType().cast<vpux::NDTypeInterface>();
    const auto canBeDoneAsEltwise = !outputType.getElementType().isa<mlir::quant::UniformQuantizedPerAxisType>();
    if (canBeDoneAsEltwise && !_useFakeSparsify) {
        rewriteSparsityOpWithEltwiseOp(rewriter, sparsifyOp.getOperation(), logCb);
    } else {
        const auto sparsityMap = createActSparsityMap(rewriter, sparsifyOp.input().getType());
        auto groupedView = rewriter.create<VPU::GroupSparseTensorOp>(sparsifyOp.getLoc(), sparsifyOp.input(),
                                                                     sparsityMap->getResult(0));
        // GroupSparseTensorOp result have new type, so cant just replaceOpWithNewOp
        sparsifyOp.output().replaceAllUsesWith(groupedView->getResult(0));
        rewriter.eraseOp(sparsifyOp);
    }
    return mlir::success();
}

//
// safeRunOnFunc
//

void LowerSparsityOpsPass::safeRunOnFunc() {
    using namespace VPU;
    using namespace VPU::NCESparsity;

    auto func = getFunction();
    auto& ctx = getContext();
    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<VPU::DesparsifyOp>();
    target.addIllegalOp<VPU::SparsifyOp>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPU::VPUDialect>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<RewriteDesparsify>(&ctx, _log);
    patterns.add<RewriteSparsify>(&ctx, _maybeFakeSparsify.getValueOr(true), _log);

    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createLowerSparsityOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createLowerSparsityOpsPass(Optional<bool> maybeFakeSparsify, Logger log) {
    return std::make_unique<LowerSparsityOpsPass>(maybeFakeSparsify, log);
}
