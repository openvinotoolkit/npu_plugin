//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/Value.h>

using namespace vpux;

namespace {

//
// DeletePerAxisQuantizationPass
//

class DeletePerAxisQuantizationPass final : public IE::DeletePerAxisQuantizationBase<DeletePerAxisQuantizationPass> {
public:
    explicit DeletePerAxisQuantizationPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    class DequantizeRewriter;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

class DeletePerAxisQuantizationPass::DequantizeRewriter final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    DequantizeRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DeletePerAxisQuantizationPass::DequantizeRewriter::matchAndRewrite(
        IE::DequantizeOp originOp, mlir::PatternRewriter& rewriter) const {
    const auto outElemType = originOp.input().getType().cast<vpux::NDTypeInterface>().getElementType();
    if (outElemType.isa<mlir::quant::UniformQuantizedType>()) {
        return mlir::failure();
    }

    auto quantOp = originOp.input().getDefiningOp<IE::QuantizeOp>();

    if (quantOp == nullptr) {
        return mlir::failure();
    }

    // check if previous operation has no fused postop
    auto producerOp = quantOp.input().getDefiningOp<IE::LayerWithPostOpInterface>();
    if (producerOp != nullptr && producerOp.getPostOp().hasValue()) {
        return mlir::failure();
    }

    rewriter.replaceOp(originOp, quantOp.input());

    return mlir::success();
}

void DeletePerAxisQuantizationPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    if (arch != VPU::ArchKind::VPUX37XX) {
        _log.trace("Deleting unused per axis quantization is for VPUX37XX only");
        return;
    }

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DequantizeRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDeletePerAxisQuantizationPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDeletePerAxisQuantizationPass(Logger log) {
    return std::make_unique<DeletePerAxisQuantizationPass>(log);
}
