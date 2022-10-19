//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <ngraph/coordinate_diff.hpp>
#include <ngraph/op/op.hpp>

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// ConvertScalarToTensorPass
//

class ConvertScalarToTensorPass final : public VPU::ConvertScalarToTensorBase<ConvertScalarToTensorPass> {
public:
    explicit ConvertScalarToTensorPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class GatherScalarConverter;
    class TopKScalarConverter;

private:
    void safeRunOnFunc() final;
};

//
// getOneDimTensor
//

vpux::VPU::ReshapeOp getOneDimTensor(mlir::Value value, mlir::Location loc, mlir::PatternRewriter& rewriter) {
    const std::array<int64_t, 1> tensorShape = {1};
    const auto tensorType = value.getType().cast<vpux::NDTypeInterface>();
    const auto newTensorType = tensorType.changeShape(ShapeRef(tensorShape));
    const auto shapeAttr = getIntArrayAttr(value.getContext(), tensorShape);
    return rewriter.create<VPU::ReshapeOp>(loc, newTensorType, value, nullptr, true, shapeAttr);
}

//
// GatherScalarConverter
//

class ConvertScalarToTensorPass::GatherScalarConverter final : public mlir::OpRewritePattern<VPU::GatherOp> {
public:
    GatherScalarConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::GatherOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::GatherOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertScalarToTensorPass::GatherScalarConverter::matchAndRewrite(
        VPU::GatherOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto newIndices = getOneDimTensor(origOp.indices(), origOp->getLoc(), rewriter);
    _log.nest().trace("New indices type '{0}'", newIndices.getType());

    rewriter.replaceOpWithNewOp<VPU::GatherOp>(origOp, origOp.getType(), origOp.input(), newIndices.output(), nullptr,
                                               origOp.axis_valueAttr(), origOp.batch_dims());
    return mlir::success();
}

//
// TopKScalarConverter
//

class ConvertScalarToTensorPass::TopKScalarConverter final : public mlir::OpRewritePattern<VPU::TopKOp> {
public:
    TopKScalarConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::TopKOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::TopKOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertScalarToTensorPass::TopKScalarConverter::matchAndRewrite(
        VPU::TopKOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto newK = getOneDimTensor(origOp.k(), origOp->getLoc(), rewriter);
    _log.nest().trace("New k type '{0}'", newK.getType());

    rewriter.replaceOpWithNewOp<VPU::TopKOp>(origOp, origOp.getResultTypes(), origOp.input(), newK.output(),
                                             origOp.axisAttr(), origOp.modeAttr(), origOp.sortAttr(),
                                             origOp.element_typeAttr());
    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertScalarToTensorPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);

    const auto rankNonZero = [](mlir::Value value) {
        return value.getType().cast<vpux::NDTypeInterface>().getRank() != 0;
    };

    target.addDynamicallyLegalOp<VPU::GatherOp>([&](VPU::GatherOp op) {
        return rankNonZero(op.indices());
    });
    target.addDynamicallyLegalOp<VPU::TopKOp>([&](VPU::TopKOp op) {
        return rankNonZero(op.k());
    });
    target.addLegalOp<VPU::ReshapeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<GatherScalarConverter>(&ctx, _log);
    patterns.insert<TopKScalarConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        _log.debug("Failed to replace indices from scalar to tensor");
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertScalarToTensorPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createConvertScalarToTensorPass(Logger log) {
    return std::make_unique<ConvertScalarToTensorPass>(log);
}
