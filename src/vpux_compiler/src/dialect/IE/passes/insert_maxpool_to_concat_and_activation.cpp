//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// GenericConverter
//

template <typename ConcreteOp>
class GenericConverter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    GenericConverter(mlir::MLIRContext* ctx, const bool useAvgPool, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _useAvgPool(useAvgPool), _log(log) {
        this->setDebugName("InsertMaxpoolToConcatLRelu::GenericConverter");
    }

private:
    mlir::LogicalResult matchAndRewrite(ConcreteOp concreteOp, mlir::PatternRewriter& rewriter) const final;

private:
    const bool _useAvgPool;
    Logger _log;
};

template <typename ConcreteOp>
mlir::LogicalResult GenericConverter<ConcreteOp>::matchAndRewrite(ConcreteOp concreteOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    auto concatOp = concreteOp.getOperand().template getDefiningOp<IE::ConcatOp>();
    if (concatOp == nullptr) {
        return mlir::failure();
    }

    const SmallVector<int64_t> poolStrides = {1, 1};
    const SmallVector<int64_t> poolKernels = {1, 1};
    const SmallVector<int64_t> pads = {0, 0};
    auto ctx = concreteOp.getContext();
    const auto padsAttr = getIntArrayAttr(ctx, pads);

    mlir::Value identityOutput = nullptr;
    if (_useAvgPool) {
        auto avgPoolOp = rewriter.create<IE::AvgPoolOp>(
                concreteOp.getLoc(), concreteOp.getOperand(), getIntArrayAttr(ctx, poolKernels),
                getIntArrayAttr(ctx, poolStrides), padsAttr, padsAttr,
                vpux::IE::RoundingTypeAttr::get(ctx, vpux::IE::RoundingType::FLOOR),
                mlir::UnitAttr::get(rewriter.getContext()), nullptr);
        identityOutput = avgPoolOp.output();
    } else {
        auto maxPoolOp = rewriter.create<IE::MaxPoolOp>(
                concreteOp.getLoc(), concreteOp.getOperand(), getIntArrayAttr(ctx, poolKernels),
                getIntArrayAttr(ctx, poolStrides), padsAttr, padsAttr,
                vpux::IE::RoundingTypeAttr::get(ctx, vpux::IE::RoundingType::FLOOR), nullptr);
        identityOutput = maxPoolOp.output();
    }

    mlir::BlockAndValueMapping mapper;
    const SmallVector<mlir::Value> inputsToMap = {identityOutput};
    mapper.map(concreteOp->getOperands(), makeArrayRef(inputsToMap));
    auto* newLayerOp = rewriter.clone(*concreteOp.getOperation(), mapper);
    rewriter.replaceOp(concreteOp, newLayerOp->getResult(0));

    return mlir::success();
}

//
// InsertMaxpoolToConcatActivationPass
//

class InsertMaxpoolToConcatActivationPass final :
        public IE::InsertMaxpoolToConcatActivationBase<InsertMaxpoolToConcatActivationPass> {
public:
    explicit InsertMaxpoolToConcatActivationPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void InsertMaxpoolToConcatActivationPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);
    const std::set<VPU::ArchKind> avgPoolTargets = {
            VPU::ArchKind::VPUX37XX,
    };
    const bool useAvgPool = avgPoolTargets.count(arch) > 0;

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GenericConverter<IE::LeakyReluOp>>(&ctx, useAvgPool, _log);
    patterns.add<GenericConverter<IE::ClampOp>>(&ctx, useAvgPool, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createInsertMaxpoolToConcatActivationPass(Logger log) {
    return std::make_unique<InsertMaxpoolToConcatActivationPass>(log);
}
