//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/passes/unroll_cluster_tiling.hpp"

namespace vpux {
namespace VPUIP {
namespace arch37xx {

//
// ClusterSWRewriter
//

class ClusterSWRewriter final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    ClusterSWRewriter(mlir::MLIRContext* ctx, mlir::ModuleOp module, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log), _ctx(ctx), _module(module) {
        setDebugName("ClusterSWRewriter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swTask, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    mlir::MLIRContext* _ctx;
    mlir::ModuleOp _module;
};

}  // namespace arch37xx
}  // namespace VPUIP
}  // namespace vpux
