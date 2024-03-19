//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Bufferization/Transforms/Bufferize.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// OneshotBufferizeVPU2VPUIPPass
//

class OneShotBufferizeVPU2VPUIPPass final : public OneShotBufferizeVPU2VPUIPBase<OneShotBufferizeVPU2VPUIPPass> {
private:
    void safeRunOnModule() final;
};

void OneShotBufferizeVPU2VPUIPPass::safeRunOnModule() {
    mlir::bufferization::OneShotBufferizationOptions options = vpux::getOneShotBufferizationOptions();
    mlir::bufferization::BufferizationStatistics statistics;
    mlir::ModuleOp moduleOp = getOperation();

    if (mlir::failed(mlir::bufferization::bufferizeOp(moduleOp, options, options.copyBeforeWrite,
                                                      /*opFilter=*/nullptr, &statistics))) {
        signalPassFailure();
        return;
    }

    mlir::bufferization::removeBufferizationAttributesInModule(getOperation());

    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    mlir::bufferization::ToMemrefOp::getCanonicalizationPatterns(patterns, &ctx);
    mlir::bufferization::ToTensorOp::getCanonicalizationPatterns(patterns, &ctx);
    if (mlir::failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOneShotBufferizeVPU2VPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createOneShotBufferizeVPU2VPUIPPass() {
    return std::make_unique<OneShotBufferizeVPU2VPUIPPass>();
}
