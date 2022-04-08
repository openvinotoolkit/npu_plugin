//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/EMU/passes.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/EMU/ops.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

//
// AdjustFQPrecisionPass
//

class AdjustFQPrecisionPass final : public EMU::AdjustFQPrecisionBase<AdjustFQPrecisionPass> {
public:
    explicit AdjustFQPrecisionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class FQPrecisionConverter;

private:
    void safeRunOnFunc() final;
};

//
// FQPrecisionConverter
//
class AdjustFQPrecisionPass::FQPrecisionConverter final : public mlir::OpRewritePattern<VPU::FakeQuantizeOp> {
public:
    FQPrecisionConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::FakeQuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

void convertToFP16(mlir::Value tensor, mlir::PatternRewriter& rewriter) {
    auto type = tensor.getType().cast<vpux::NDTypeInterface>();
    const auto elementType = type.getElementType();
    auto parentConstOp = tensor.getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_UNLESS(parentConstOp != nullptr, "Got non constant parameters for FakeQuantize");

    if (elementType.isF16()) {
        return;
    }

    const auto newElementType = mlir::Float16Type::get(elementType.getContext());
    const auto newTensorType = type.changeElemType(newElementType);
    const auto newConstAttr = parentConstOp.contentAttr().convertElemType(newElementType);
    auto newTensor = rewriter.create<Const::DeclareOp>(parentConstOp.getLoc(), newTensorType, newConstAttr).output();
    parentConstOp.replaceAllUsesWith(newTensor);
}

mlir::LogicalResult AdjustFQPrecisionPass::FQPrecisionConverter::matchAndRewrite(
        VPU::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const {
    rewriter.startRootUpdate(origOp);
    convertToFP16(origOp.input_low(), rewriter);
    convertToFP16(origOp.input_high(), rewriter);
    convertToFP16(origOp.output_low(), rewriter);
    convertToFP16(origOp.output_high(), rewriter);
    rewriter.finalizeRootUpdate(origOp);
    return mlir::success();
}

//
// safeRunOnFunc
//

void AdjustFQPrecisionPass::safeRunOnFunc() {
    auto& ctx = getContext();

    auto checkFP16Dtype = [](mlir::Value tensor) {
        return tensor.getType().cast<vpux::NDTypeInterface>().getElementType().isF16();
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<Const::DeclareOp>();
    target.addDynamicallyLegalOp<VPU::FakeQuantizeOp>([checkFP16Dtype](VPU::FakeQuantizeOp op) {
        return checkFP16Dtype(op.input_low()) && checkFP16Dtype(op.input_high()) && checkFP16Dtype(op.output_low()) &&
               checkFP16Dtype(op.output_high());
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FQPrecisionConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createAdjustFQPrecisionPass
//

std::unique_ptr<mlir::Pass> vpux::EMU::createAdjustFQPrecisionPass(Logger log) {
    return std::make_unique<AdjustFQPrecisionPass>(log);
}
