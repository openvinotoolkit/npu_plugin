//
// Copyright 2021 Intel Corporation.
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

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IERT/passes.hpp"

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// FuseActivationsPass
//

class FuseActivationsPass final : public FuseActivationsBase<FuseActivationsPass> {
public:
    FuseActivationsPass(Logger log);

public:
    class ReLURewrite;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

FuseActivationsPass::FuseActivationsPass(Logger log): _log(log) {
    _log.setName(Base::getArgumentName());
}

//
// ReLURewrite
//

class FuseActivationsPass::ReLURewrite final : public mlir::OpRewritePattern<IERT::ReLUOp> {
public:
    ReLURewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IERT::ReLUOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::ReLUOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseActivationsPass::ReLURewrite::matchAndRewrite(IERT::ReLUOp origOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    // ReLU input is expected to come from reorder task which repacks NCE2 output from NHWC to NCHW
    // FIXME reorder will be skipped when adjust layouts pass is added to the hardware pipeline
    // When it happens, don't forget to remove this reorder here
    auto nce_out_reorder = origOp.input().getDefiningOp<IERT::ReorderOp>();
    if (!nce_out_reorder) {
        return matchFailed(rewriter, origOp, "ReLU input does not look like NCE task output");
    }
    // Reorder input comes from NNDMA task which transmits NCE2 result from CMX to DDR
    auto cmx_to_ddr = nce_out_reorder.input().getDefiningOp<IERT::CopyOp>();
    if (!cmx_to_ddr) {
        return matchFailed(rewriter, origOp, "ReLU input does not look like NCE task output");
    }
    auto nce_cluster_task = cmx_to_ddr.input().getDefiningOp<VPUIP::NCEClusterTaskOp>();
    if (!nce_cluster_task) {
        return matchFailed(rewriter, origOp, "ReLU input is not connected to NCE task output");
    }

    auto ctx = getContext();
    auto relu_ppe_attr = vpux::VPUIP::PPELayerTypeAttr::get(ctx, VPUIP::PPELayerType::LRELU);
    nce_cluster_task.fixed_ppe_taskAttr(relu_ppe_attr);

    rewriter.replaceOp(origOp, nce_out_reorder->getResults());

    return mlir::success();
}

void FuseActivationsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.insert<ReLURewrite>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::createFuseActivationsPass(Logger log) {
    return std::make_unique<FuseActivationsPass>(log);
}
