//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/logger.hpp"

#include <memory>

namespace vpux {
namespace VPUIP {

//
// ClusterNCEBaseRewriter
//

class ClusterNCEBaseRewriter : public mlir::OpRewritePattern<VPUIP::NCEClusterTaskOp> {
public:
    ClusterNCEBaseRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::NCEClusterTaskOp>(ctx), _log(log), _ctx(ctx) {
        _cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::NCEClusterTaskOp nceTask, mlir::PatternRewriter& rewriter) const final;

protected:
    virtual void getOutputBuffers(SmallVector<mlir::Value>& parentOutputBuffs, SmallVector<mlir::Value>& outputBuffs,
                                  SmallVector<mlir::Value>& parentOutputSparsityMap,
                                  SmallVector<mlir::Value>& outputSparsityMapBuffs, mlir::Location loc,
                                  VPUIP::NCEClusterTilingOp clusterOp, VPUIP::NCEClusterTaskOp nceTask,
                                  const int64_t numClusters, mlir::PatternRewriter& rewriter) const = 0;

    virtual void getInputBuffers(SmallVector<mlir::Value>& parentInputBuffs, SmallVector<mlir::Value>& inputBuffs,
                                 SmallVector<mlir::Value>& parentInputSparsityMap,
                                 SmallVector<mlir::Value>& inputSparsityMapBuffs,
                                 SmallVector<mlir::Value>& parentInputSETable,
                                 SmallVector<mlir::Value>& inputSETableBuffs, mlir::Location loc,
                                 VPUIP::NCEClusterTilingOp clusterOp, VPUIP::NCEClusterTaskOp nceTask,
                                 const int64_t numClusters, mlir::PatternRewriter& rewriter) const = 0;

    virtual SmallVector<mlir::Value> getWeightsBuffers(mlir::Location loc, VPUIP::NCEClusterTilingOp clusterOp,
                                                       VPUIP::NCEClusterTaskOp nceTask, const int64_t numClusters,
                                                       mlir::PatternRewriter& rewriter) const;

    virtual mlir::UnitAttr isSegmentedNCETask(VPUIP::DistributedBufferType inputType) const = 0;

private:
    SmallVector<mlir::IntegerAttr> getOutChannelOffsets(VPUIP::NCEClusterTaskOp nceTask,
                                                        VPUIP::DistributedBufferType inType,
                                                        VPUIP::DistributedBufferType outType) const;

protected:
    Logger _log;
    mlir::MLIRContext* _ctx;
    mlir::FlatSymbolRefAttr _cmxNameAttr;
};

//
// ClusterDMARewriter
//

class ClusterDMARewriter final : public mlir::OpRewritePattern<VPUIP::NNDMAOp> {
public:
    ClusterDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log)
            : mlir::OpRewritePattern<VPUIP::NNDMAOp>(ctx), _log(log), _ctx(ctx), _dmaPortCount(dmaPortCount) {
        setDebugName("ClusterDMARewriter");

        _cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::NNDMAOp nndmaOp, mlir::PatternRewriter& rewriter) const final;

private:
    void unrollSegmentedOrOverlapped(mlir::Location loc, VPUIP::NCEClusterTilingOp clusterOp, VPURT::TaskOp vpurtTask,
                                     VPUIP::DistributedBufferType distributedType,
                                     mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
    mlir::MLIRContext* _ctx;
    int64_t _dmaPortCount;
    mlir::FlatSymbolRefAttr _cmxNameAttr;
};

}  // namespace VPUIP
}  // namespace vpux
