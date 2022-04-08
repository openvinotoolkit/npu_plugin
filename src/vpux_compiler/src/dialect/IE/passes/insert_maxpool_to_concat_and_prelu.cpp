//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// GenericConverter
//

class GenericConverter final : public mlir::OpRewritePattern<IE::LeakyReluOp> {
public:
    GenericConverter(mlir::MLIRContext* ctx, const bool useAvgPool, Logger log)
            : mlir::OpRewritePattern<IE::LeakyReluOp>(ctx), _useAvgPool(useAvgPool), _log(log) {
        this->setDebugName("InsertMaxpoolToConcatLRelu::GenericConverter");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::LeakyReluOp leakyReluOp, mlir::PatternRewriter& rewriter) const final;

private:
    const bool _useAvgPool;
    Logger _log;
};

mlir::LogicalResult GenericConverter::matchAndRewrite(IE::LeakyReluOp leakyReluOp,
                                                      mlir::PatternRewriter& rewriter) const {
    auto concatOp = leakyReluOp.getOperand().getDefiningOp<IE::ConcatOp>();
    if (concatOp == nullptr) {
        return mlir::failure();
    }

    const SmallVector<int64_t> poolStrides = {1, 1};
    const SmallVector<int64_t> poolKernels = {1, 1};
    const SmallVector<int64_t> pads = {0, 0};
    auto ctx = leakyReluOp.getContext();
    const auto padsAttr = getIntArrayAttr(ctx, pads);

    mlir::Value identityOutput = nullptr;
    if (_useAvgPool) {
        auto avgPoolOp = rewriter.create<IE::AvgPoolOp>(
                leakyReluOp.getLoc(), leakyReluOp.getOperand(), getIntArrayAttr(ctx, poolKernels),
                getIntArrayAttr(ctx, poolStrides), padsAttr, padsAttr,
                vpux::IE::RoundingTypeAttr::get(ctx, vpux::IE::RoundingType::FLOOR),
                mlir::UnitAttr::get(rewriter.getContext()), nullptr);
        identityOutput = avgPoolOp.output();
    } else {
        auto maxPoolOp = rewriter.create<IE::MaxPoolOp>(
                leakyReluOp.getLoc(), leakyReluOp.getOperand(), getIntArrayAttr(ctx, poolKernels),
                getIntArrayAttr(ctx, poolStrides), padsAttr, padsAttr,
                vpux::IE::RoundingTypeAttr::get(ctx, vpux::IE::RoundingType::FLOOR), nullptr);
        identityOutput = maxPoolOp.output();
    }

    rewriter.replaceOpWithNewOp<IE::LeakyReluOp>(leakyReluOp, identityOutput, leakyReluOp.negative_slopeAttr());

    return mlir::success();
}

//
// InsertMaxpoolToConcatLReluPass
//

class InsertMaxpoolToConcatLReluPass final : public IE::InsertMaxpoolToConcatLReluBase<InsertMaxpoolToConcatLReluPass> {
public:
    explicit InsertMaxpoolToConcatLReluPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void InsertMaxpoolToConcatLReluPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);
    const std::set<VPU::ArchKind> avgPoolTargets = {
            VPU::ArchKind::VPUX37XX,
    };
    const bool useAvgPool = avgPoolTargets.count(arch) > 0;

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GenericConverter>(&ctx, useAvgPool, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createInsertMaxpoolToConcatLReluPass(Logger log) {
    return std::make_unique<InsertMaxpoolToConcatLReluPass>(log);
}
