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
#include "vpux/compiler/dialect/EMU/passes.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/EMU/ops.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
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
class AdjustFQPrecisionPass::FQPrecisionConverter final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    FQPrecisionConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

void convertToFP16(mlir::Value tensor, mlir::PatternRewriter& rewriter)
{
    auto type = tensor.getType().cast<mlir::RankedTensorType>();
    const auto elementType = type.getElementType();
    auto parentConstOp = tensor.getDefiningOp<Const::DeclareOp>();
    if (parentConstOp == nullptr)
        VPUX_THROW("Got non constant parameters for FakeQuantize");

    if (!elementType.isF16())
    {
        const auto newElementType = mlir::Float16Type::get(elementType.getContext());
        const auto newTensorType = type.clone(newElementType);
        const auto newConstAttr = parentConstOp.contentAttr().convertElemType(newElementType);
        auto newTensor = rewriter.create<Const::DeclareOp>(parentConstOp.getLoc(), newTensorType, newConstAttr).output();
        parentConstOp.replaceAllUsesWith(newTensor);
    }
}

mlir::LogicalResult AdjustFQPrecisionPass::FQPrecisionConverter::matchAndRewrite(IE::FakeQuantizeOp origOp,
                                    mlir::PatternRewriter& rewriter) const {
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
        return tensor.getType().cast<mlir::RankedTensorType>().getElementType().isF16();
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<Const::DeclareOp>();
    target.addDynamicallyLegalOp<IE::FakeQuantizeOp>([checkFP16Dtype](IE::FakeQuantizeOp op)
        {
            return checkFP16Dtype(op.input_low()) &&
                checkFP16Dtype(op.input_high()) &&
                checkFP16Dtype(op.output_low()) &&
                checkFP16Dtype(op.output_high());
        });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<FQPrecisionConverter>(&ctx, _log);

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
