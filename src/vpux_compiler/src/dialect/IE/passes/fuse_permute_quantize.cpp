//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/enums.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

// ======================================================================================
// FusePermuteQuantizeRewrite
//   FusePermuteQuantizeRewrite -> [Reorder -> Add -> QuantizeCastOp] -> PermuteQuantize

class FusePermuteQuantizeRewrite final : public mlir::OpRewritePattern<IE::QuantizeCastOp> {
public:
    FusePermuteQuantizeRewrite(mlir::MLIRContext* ctx, const bool dpuOnly, Logger log)
            : mlir::OpRewritePattern<IE::QuantizeCastOp>(ctx, benefitHigh), _dpuOnly(dpuOnly), _log(log) {
        setDebugName("FusePermuteQuantizeRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeCastOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isCompatibleWithDPU(mlir::Type addInput, mlir::Type addOutput) const;
    const bool _dpuOnly;
    Logger _log;
};

mlir::LogicalResult FusePermuteQuantizeRewrite::matchAndRewrite(IE::QuantizeCastOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // check patern
    auto opAdd = origOp.input().getDefiningOp<IE::AddOp>();
    if (opAdd == nullptr) {
        return mlir::failure();
    }
    auto opReorder = opAdd.input1().getDefiningOp<IE::ReorderOp>();
    if (opReorder == nullptr) {
        return mlir::failure();
    }

    // check just 1 child for linked patern
    for (auto user : llvm::make_early_inc_range(opReorder.getResult().getUsers())) {
        if (user != opAdd) {
            return mlir::failure();
        }
    }
    if (!opAdd.getResult().hasOneUse()) {
        return mlir::failure();
    }

    // check if Add is used for quantize
    if (opAdd.input1() != opAdd.input2()) {
        return mlir::failure();
    }
    const auto inType = opAdd.input1().getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto outType = opAdd.output().getType().cast<vpux::NDTypeInterface>().getElementType();
    if (!(inType.isF16() && outType.isa<mlir::quant::QuantizedType>())) {
        return mlir::failure();
    }

    // check uniform quantize
    const auto qType = outType.cast<mlir::quant::QuantizedType>();
    if (!qType.isa<mlir::quant::UniformQuantizedType>()) {
        return mlir::failure();
    }

    // check if reorder will not be removed
    auto inOrder = DimsOrder::fromValue(opReorder.input());
    auto outOrder = DimsOrder::fromValue(opReorder.output());
    if (inOrder == outOrder) {
        return mlir::failure();
    }
    // check and add pass for verified orders and scenarios
    if (!((inOrder == DimsOrder::NCHW) && (outOrder == DimsOrder::NHWC))) {
        return mlir::failure();
    }

    //
    const auto iExpType = opReorder.input().getType().cast<vpux::NDTypeInterface>();
    const auto oExpType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    if (!((iExpType.getRank() == 4) && (oExpType.getRank() == 4))) {
        return mlir::failure();
    }

    // experiments show that shave is far more performant when C == 1, C == 3 or C == 4 than DMA-MemPermute
    if ((iExpType.getShape()[Dims4D::Act::C] != 3) && (iExpType.getShape()[Dims4D::Act::C] != 1) &&
        (iExpType.getShape()[Dims4D::Act::C] != 4)) {
        return mlir::failure();
    }
    if (iExpType.getShape()[Dims4D::Act::N] != 1) {
        return mlir::failure();
    }

    // If subgraph is SpaceToDepth->Reorder->Add(Quantize),
    // conversion to SpaceToDepthDMA->Swkernel(PermuteQuantize) is much slower than
    // conversion to SpaceToDepth->MemPermute(Reorder)->NCEEltwise(Quantize) which
    // later will be fused as SpaceToDepthDMA->NCEEltwise(Quantize).
    // In this case, disable fuse of Reorder and Add as PermuteQuantize here.
    if (auto s2dOp = opReorder.input().getDefiningOp<IE::SpaceToDepthOp>()) {
        return mlir::failure();
    }

    if (_dpuOnly && !isCompatibleWithDPU(opAdd.input1().getType(), opAdd.output().getType())) {
        return mlir::failure();
    }

    auto memPermAttr = mlir::AffineMapAttr::get(getPermutationFromOrders(inOrder, outOrder, origOp->getContext()));
    SmallVector<int64_t> noPadBeginEnd(inOrder.numDims(), 0);
    const auto& ctx = origOp.getContext();
    // Get target quant type from IE.Add, not from IE.QuantizeCast.
    // QuantizeToAddRewriter multiplies output scale by 2. It is necessary to cancel out this factor.
    const auto permQuantOutType = rescaleUniformQuantizedType(opAdd.output().getType(), 0.5);
    const auto permQuantElemType = permQuantOutType.cast<vpux::NDTypeInterface>().getElementType();
    const auto dstElemTypeAttr = mlir::TypeAttr::get(permQuantElemType);
    const auto permQuantLoc = appendLoc(origOp->getLoc(), "PermuteQuantize");
    auto permuteQuantizeOp = rewriter.create<IE::PermuteQuantizeOp>(
            permQuantLoc, permQuantOutType, opReorder.input(), opReorder.dstOrderAttr(), memPermAttr, dstElemTypeAttr,
            getIntArrayAttr(ctx, noPadBeginEnd), getIntArrayAttr(ctx, noPadBeginEnd));

    // IE.PermuteQuantize must have quantization parameters from the original IE.Quantize operation.
    // In some cases IE.QuantizeCast which follows IE.Add can contain dstElemType which differs from that IE.Quantize.
    // For example, one IE.QuantizeCast may appear after IE.FakeQuantize gets split into:
    // IE.Quantize qType1 -> IE.QuantizeCast qType2 -> IE.Dequantize
    // Another IE.QuantizeCast will be inserted into graph after QuantizeToAddRewriter:
    // IE.Add qType0 -> IE.QuantizeCast qType1 -> IE.QuantizeCast qType2 -> IE.Dequantize
    // Such chain of two consecutive IE.QuantizeCast will be fused into one:
    // IE.Add qType0 -> IE.QuantizeCast qType2 -> IE.Dequantize
    // In that case, qType1 must be set for IE.PermuteQuantize.
    // IE.QuantizeCast to qType2 must remain in the graph to maintain the integrity:
    // IE.PermuteQuantize qType1 -> IE.QuantizeCast qType2 -> IE.Dequantize
    auto quantCast =
            rewriter.create<IE::QuantizeCastOp>(origOp->getLoc(), permuteQuantizeOp.output(), origOp.dstElemTypeAttr());

    rewriter.replaceOp(origOp, quantCast.output());

    return mlir::success();
}

bool FusePermuteQuantizeRewrite::isCompatibleWithDPU(mlir::Type addInput, mlir::Type addOutput) const {
    auto inType = addInput.cast<vpux::NDTypeInterface>();
    auto outType = addOutput.cast<vpux::NDTypeInterface>();
    const auto inElemType = inType.getElementType();
    if (!inElemType.isF16()) {
        return false;
    }
    const auto outElemType = outType.getElementType();
    if (!outElemType.isF16() && !outElemType.isa<mlir::quant::UniformQuantizedType>()) {
        return false;
    }
    const ShapeRef inShape = inType.getShape();
    const auto inAlignment = VPU::NCEInvariant::getAlignment(inElemType);
    if (!IE::isODUPermuteEffectiveForShape(inShape, inAlignment)) {
        return false;
    }
    const ShapeRef outShape = outType.getShape();
    const auto outAlignment = VPU::NCEInvariant::getAlignment(outElemType);
    if (!IE::isODUPermuteEffectiveForShape(outShape, outAlignment)) {
        return false;
    }

    return true;
}

//
// FusePermuteQuantizePass
//

class FusePermuteQuantizePass final : public IE::FusePermuteQuantizeBase<FusePermuteQuantizePass> {
public:
    explicit FusePermuteQuantizePass(const bool dpuOnly, Logger log): _dpuOnly(dpuOnly), _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    bool _dpuOnly;
    Logger _log;
};

mlir::LogicalResult FusePermuteQuantizePass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (dpuOnly.hasValue()) {
        _dpuOnly = dpuOnly.getValue();
    }

    return mlir::success();
}

void FusePermuteQuantizePass::safeRunOnFunc() {
    // TODO: #70647

    auto& ctx = getContext();
    auto func = getOperation();

    // dpuOnly flag means that target platform supports only DPU implementation of PermuteQuantize.
    // In that case PermuteQuantize fusion has some limitations:
    // 1. Only NCHW to NHWC permutation is supported
    // 2. Only float16 inputs and quantized outputs are supported.
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FusePermuteQuantizeRewrite>(&ctx, _dpuOnly, _log);

    mlir::ConversionTarget target(ctx);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createFusePermuteQuantizePass
//
std::unique_ptr<mlir::Pass> vpux::IE::createFusePermuteQuantizePass(const bool dpuOnly, Logger log) {
    return std::make_unique<FusePermuteQuantizePass>(dpuOnly, log);
}
