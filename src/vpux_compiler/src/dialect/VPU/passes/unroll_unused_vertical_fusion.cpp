//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>

using namespace vpux;
using namespace VPU;

namespace {

//
// VerticalFusionUnrollRewriter
//

class VerticalFusionUnrollRewriter final : public mlir::OpRewritePattern<VPU::VerticalFusionOp> {
public:
    VerticalFusionUnrollRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::VerticalFusionOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPU::VerticalFusionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult VerticalFusionUnrollRewriter::matchAndRewrite(VPU::VerticalFusionOp vfOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    mlir::BlockAndValueMapping mapper;
    for (auto arg : vfOp.getBody()->getArguments()) {
        mapper.map(arg, vfOp.getOperand(arg.getArgNumber()));
    }

    mlir::Operation* clonedOp = nullptr;
    for (auto& op : vfOp.getBody()->without_terminator()) {
        clonedOp = rewriter.clone(op, mapper);
        if (mlir::isa<VPU::VerticalFusionOpInterface>(clonedOp)) {
            clonedOp->setAttr(vfOp.tilingStrategyAttrName(), vfOp.tilingStrategyAttr());
        }
        mapper.map(op.getResult(0), clonedOp->getResult(0));
    }

    rewriter.replaceOp(vfOp, clonedOp->getResult(0));

    return mlir::success();
}

//
// UnrollUnusedVerticalFusionRegionPass
//

class UnrollUnusedVerticalFusionRegionPass final :
        public UnrollUnusedVerticalFusionRegionBase<UnrollUnusedVerticalFusionRegionPass> {
public:
    explicit UnrollUnusedVerticalFusionRegionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnModule
//

void UnrollUnusedVerticalFusionRegionPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    const auto isLegalVF = [&](VPU::VerticalFusionOp op) {
        auto vfOps = op.body().front().getOps<VPU::VerticalFusionOpInterface>();
        if (std::distance(vfOps.begin(), vfOps.end()) == 1) {
            return false;
        }

        const auto tilingInfo = parseIntArrayAttr<int64_t>(op.tilingStrategy());
        const auto hasTiling = llvm::any_of(tilingInfo, [](int64_t i) {
            return i != 1;
        });
        if (!hasTiling) {
            return false;
        }
        return true;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<VPU::VerticalFusionOp>(isLegalVF);
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPU::VPUDialect>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<VerticalFusionUnrollRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollUnusedVerticalFusionRegionPass
//

std::unique_ptr<mlir::Pass> VPU::createUnrollUnusedVerticalFusionRegionPass(Logger log) {
    return std::make_unique<UnrollUnusedVerticalFusionRegionPass>(log);
}
