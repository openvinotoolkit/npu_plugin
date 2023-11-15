//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include "vpux/compiler/core/act_profiling.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/sw_utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <numeric>

using namespace vpux;

namespace {

bool hasMultiSwKernelRun(VPUIP::SwKernelOp swKernelOp) {
    auto swKernelRun = swKernelOp.body().getOps<VPUIP::SwKernelRun>();
    return std::distance(swKernelRun.begin(), swKernelRun.end()) > 1;
}

SmallVector<mlir::Value> getOuterMostMappingOperand(VPUIP::SwKernelRun swKernelRun) {
    auto swKernelOp = swKernelRun->getParentOfType<VPUIP::SwKernelOp>();
    VPUX_THROW_WHEN(swKernelOp == nullptr, "Cannot find VPUIP.SwKernelOp at '{0}'", swKernelRun->getLoc());

    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
    auto isClusterTilingApplied = clusterTilingOp != nullptr;
    SmallVector<mlir::Value> outerMostOperands;

    for (auto operand : swKernelRun->getOperands()) {
        auto blockArg = operand.dyn_cast<mlir::BlockArgument>();
        VPUX_THROW_WHEN(blockArg == nullptr, "Matching argument was not identified");
        auto outerOperand = swKernelOp->getOperand(blockArg.getArgNumber());
        if (isClusterTilingApplied) {
            blockArg = outerOperand.dyn_cast<mlir::BlockArgument>();
            VPUX_THROW_WHEN(blockArg == nullptr, "Matching argument was not identified");
            outerOperand = clusterTilingOp->getOperand(blockArg.getArgNumber());
        }
        outerMostOperands.push_back(outerOperand);
    }
    return outerMostOperands;
}

bool isOperandFromList(mlir::ValueRange rangeList, mlir::Value operand) {
    return llvm::find(rangeList, operand) != rangeList.end();
}

//
// SwKernelRewriterBase
//

class SwKernelRewriterBase : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    SwKernelRewriterBase(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log), _ctx(ctx) {
        setDebugName("SwKernelRewriterBase");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const final;

    virtual bool needUnroll(VPUIP::SwKernelOp swKernelOp) const = 0;
    virtual VPURT::TaskOp createNewTaskOp(VPUIP::SwKernelOp swKernelOp, VPUIP::SwKernelRun swKernelRun,
                                          VPURT::TaskOp origTaskOp, mlir::PatternRewriter& rewriter,
                                          size_t index) const = 0;

protected:
    Logger _log;
    mlir::MLIRContext* _ctx;
};

mlir::LogicalResult SwKernelRewriterBase::matchAndRewrite(VPUIP::SwKernelOp swKernelOp,
                                                          mlir::PatternRewriter& rewriter) const {
    if (!needUnroll(swKernelOp)) {
        return mlir::failure();
    }

    auto vpurtTask = swKernelOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");
    auto cycleBeginAttr = vpurtTask->getAttr(cycleBegin);
    auto cycleEndAttr = vpurtTask->getAttr(cycleEnd);

    rewriter.setInsertionPointAfter(vpurtTask);
    auto swKernelRunList = swKernelOp.body().getOps<VPUIP::SwKernelRun>();
    for (auto swKernelRunTuple : swKernelRunList | indexed) {
        auto newTaskOp =
                createNewTaskOp(swKernelOp, swKernelRunTuple.value(), vpurtTask, rewriter, swKernelRunTuple.index());

        if (cycleBeginAttr) {
            newTaskOp->setAttr(cycleBegin, cycleBeginAttr);
        }
        if (cycleEndAttr) {
            newTaskOp->setAttr(cycleEnd, cycleEndAttr);
        }
        _log.trace("create new task op: {0}", newTaskOp);
    }
    rewriter.eraseOp(vpurtTask);
    return mlir::success();
}

//
// SwKernelRewriter
//

class SwKernelRewriter final : public SwKernelRewriterBase {
public:
    SwKernelRewriter(mlir::MLIRContext* ctx, Logger log): SwKernelRewriterBase(ctx, log) {
        setDebugName("SwKernelRewriter");
    }

    bool needUnroll(VPUIP::SwKernelOp swKernelOp) const override;
    VPURT::TaskOp createNewTaskOp(VPUIP::SwKernelOp swKernelOp, VPUIP::SwKernelRun swKernelRun,
                                  VPURT::TaskOp origTaskOp, mlir::PatternRewriter& rewriter,
                                  size_t index) const override;
};

bool SwKernelRewriter::needUnroll(VPUIP::SwKernelOp swKernelOp) const {
    if (mlir::isa<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp())) {
        return false;
    }
    auto hasMultiSwKernelRunFlag = hasMultiSwKernelRun(swKernelOp);
    if (!hasMultiSwKernelRunFlag && swKernelOp.profiling_data() == nullptr) {
        // SW task is not going to be unrolled, update its name to indicate
        // its cluster and tile index to align with name structure in case of unrolling
        auto oldLoc = swKernelOp->getLoc();
        if (stringifyLocation(oldLoc).find("/tile_") == std::string::npos &&
            stringifyLocation(oldLoc).find("?tile_") == std::string::npos) {
            auto newLoc = appendLoc(oldLoc, "tile_0");
            swKernelOp->setLoc(appendLoc(newLoc, "cluster_0"));
        }
    }
    return hasMultiSwKernelRunFlag;
}

VPURT::TaskOp SwKernelRewriter::createNewTaskOp(VPUIP::SwKernelOp swKernelOp, VPUIP::SwKernelRun swKernelRun,
                                                VPURT::TaskOp origTaskOp, mlir::PatternRewriter& rewriter,
                                                size_t index) const {
    auto opLoc = swKernelOp->getLoc();
    auto outerOperand = getOuterMostMappingOperand(swKernelRun);
    auto iter = llvm::find_if(outerOperand, [&](auto operand) {
        return isOperandFromList(swKernelOp.output_buffs(), operand);
    });
    VPUX_THROW_WHEN(iter == outerOperand.end(), "Cannot find operand for output buffer at '{0}'", opLoc);

    auto outBufferStartIndex = std::distance(outerOperand.begin(), iter);
    auto newInputs = SmallVector<mlir::Value>(outerOperand.begin(), outerOperand.begin() + outBufferStartIndex);
    auto newOutBuffers = SmallVector<mlir::Value>(outerOperand.begin() + outBufferStartIndex, outerOperand.end());

    mlir::Value newProfilingBuffer = nullptr;
    if (auto profilingBuffer = swKernelOp.profiling_data()) {
        // In case task has profiling buffer as SwKernelOp operand, each individual SwKernelRun will be given
        // its own chunk of this buffer. Buffer size was properly prepared by act-shave-profiling pass with
        // enough space for each SwKernelRun inside
        opLoc = vpux::getUpdatedActShaveProfilingLoc(opLoc, index);
        auto profilingBufferDecl = profilingBuffer.getDefiningOp<VPURT::DeclareBufferOp>();
        auto profilingBufferNDType = profilingBuffer.getType().cast<vpux::NDTypeInterface>();

        int64_t numEl =
                VPUIP::HW_ACT_SHAVE_PROFILING_SIZE_BYTES / vpux::Byte(profilingBufferNDType.getElemTypeSize()).count();

        const auto newMemType = profilingBufferNDType.changeShape({numEl});
        auto newProfilingBufferDecl = rewriter.create<VPURT::DeclareBufferOp>(
                profilingBufferDecl->getLoc(), newMemType, profilingBufferDecl.getSectionAttr(),
                profilingBufferDecl.getSectionIndexAttr(),
                getIntAttr(_ctx,
                           profilingBufferDecl.getByteOffset() + VPUIP::HW_ACT_SHAVE_PROFILING_SIZE_BYTES * index),
                nullptr);

        newProfilingBuffer = newProfilingBufferDecl.getBuffer();
    } else {
        opLoc = appendLoc(opLoc, "tile_{0}", index);
        opLoc = appendLoc(opLoc, "cluster_0");
    }

    auto newSwKernelOp = VPURT::wrapIntoTaskOp<VPUIP::SwKernelOp>(
            rewriter, origTaskOp.getWaitBarriers(), origTaskOp.getUpdateBarriers(), opLoc, newInputs, newOutBuffers,
            newProfilingBuffer, swKernelOp.kernelFunctionAttr(), swKernelOp.tileIndexAttr());
    VPUIP::initSwKernel(newSwKernelOp, swKernelRun, _log);
    return newSwKernelOp->getParentOfType<VPURT::TaskOp>();
}

//
// ClusterSwKernelRewriter
//

class ClusterSwKernelRewriter final : public SwKernelRewriterBase {
public:
    ClusterSwKernelRewriter(mlir::MLIRContext* ctx, Logger log): SwKernelRewriterBase(ctx, log) {
        setDebugName("ClusterSwKernelRewriter");
    }
    bool needUnroll(VPUIP::SwKernelOp swKernelOp) const override;
    VPURT::TaskOp createNewTaskOp(VPUIP::SwKernelOp swKernelOp, VPUIP::SwKernelRun swKernelRun,
                                  VPURT::TaskOp origTaskOp, mlir::PatternRewriter& rewriter,
                                  size_t index) const override;
};

bool ClusterSwKernelRewriter::needUnroll(VPUIP::SwKernelOp swKernelOp) const {
    if (!mlir::isa<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp())) {
        return false;
    }
    auto hasMultiSwKernelRunFlag = hasMultiSwKernelRun(swKernelOp);
    if (!hasMultiSwKernelRunFlag && swKernelOp.profiling_data() == nullptr) {
        // SW task is not going to be unrolled, update its name to indicate
        // its tile index to align with name structure in case of unrolling
        auto oldLoc = swKernelOp->getLoc();
        if (stringifyLocation(oldLoc).find("/tile_") == std::string::npos) {
            swKernelOp->setLoc(appendLoc(oldLoc, "tile_0"));
        }
    }
    return hasMultiSwKernelRunFlag;
}

VPURT::TaskOp ClusterSwKernelRewriter::createNewTaskOp(VPUIP::SwKernelOp swKernelOp, VPUIP::SwKernelRun swKernelRun,
                                                       VPURT::TaskOp origTaskOp, mlir::PatternRewriter& rewriter,
                                                       size_t index) const {
    auto opLoc = swKernelOp->getLoc();
    auto outerOperand = getOuterMostMappingOperand(swKernelRun);
    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(swKernelOp->getParentOp());
    auto iter = llvm::find_if(outerOperand, [&](auto operand) {
        return isOperandFromList(clusterTilingOp.output_buffs(), operand);
    });
    VPUX_THROW_WHEN(iter == outerOperand.end(), "Cannot find operand for output buffer at '{0}'", opLoc);
    auto outBufferStartIndex = std::distance(outerOperand.begin(), iter);

    auto profilingInnerOperand = swKernelOp.profiling_data();
    bool isProfEnabled = (profilingInnerOperand != nullptr);

    if (isProfEnabled) {
        // In case task has profiling buffer as SwKernelOp operand, each individual SwKernelRun will be given
        // its own chunk of this buffer. Buffer size was properly prepared by act-shave-profiling pass with
        // enough space for each SwKernelRun inside
        const auto profBlockArg = profilingInnerOperand.cast<mlir::BlockArgument>();
        auto profilingOuterOperand = clusterTilingOp->getOperand(profBlockArg.getArgNumber());

        opLoc = vpux::getUpdatedActShaveProfilingLoc(opLoc, index);

        auto profilingBufferDecl = profilingOuterOperand.getDefiningOp<VPURT::DeclareBufferOp>();

        auto distProfBufType = profilingOuterOperand.getType().dyn_cast<VPUIP::DistributedBufferType>();
        VPUX_THROW_WHEN(distProfBufType == nullptr, "Porfiling buffer is not of distributed type - '{0}'",
                        profilingOuterOperand.getType());

        int64_t numEl = distProfBufType.getDistribution().getNumClusters().getInt() *
                        VPUIP::HW_ACT_SHAVE_PROFILING_SIZE_BYTES /
                        vpux::Byte(distProfBufType.getElemTypeSize()).count();
        int64_t offset = profilingBufferDecl.getByteOffset() + VPUIP::HW_ACT_SHAVE_PROFILING_SIZE_BYTES * index;

        const auto newMemType = distProfBufType.changeShape({numEl});
        auto newProfilingBufferDecl = rewriter.create<VPURT::DeclareBufferOp>(
                profilingBufferDecl->getLoc(), newMemType, profilingBufferDecl.getSectionAttr(),
                profilingBufferDecl.getSectionIndexAttr(), getIntAttr(_ctx, offset), nullptr);

        outerOperand.push_back(newProfilingBufferDecl.getBuffer());
    } else {
        opLoc = appendLoc(opLoc, "tile_{0}", index);
    }

    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange operands) {
        SmallVector<mlir::Value> inputs(operands.begin(), operands.begin() + outBufferStartIndex);
        auto outputEndIt = operands.end();
        mlir::Value profOutput = nullptr;
        if (isProfEnabled) {
            // profiling is last operand
            profOutput = operands.back();
            outputEndIt--;
        }
        SmallVector<mlir::Value> outputs(operands.begin() + outBufferStartIndex, outputEndIt);
        auto newSwKernelTask =
                builder.create<VPUIP::SwKernelOp>(loc, inputs, outputs, profOutput, swKernelOp.kernelFunction(),
                                                  swKernelOp.tileIndexAttr(), swKernelOp.stridesAttr());
        VPUIP::initSwKernel(newSwKernelTask, swKernelRun, _log);
    };

    SmallVector<mlir::Type> resultTypes;
    for (; iter != outerOperand.end(); iter++) {
        resultTypes.push_back(iter->getType());
    }
    auto newClusterTilingOp = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTilingOp>(rewriter, origTaskOp.getWaitBarriers(),
                                                                               origTaskOp.getUpdateBarriers(), opLoc,
                                                                               resultTypes, outerOperand, bodyBuilder);
    return newClusterTilingOp->getParentOfType<VPURT::TaskOp>();
}

//
// UnrollSwKernelPass
//

class UnrollSwKernelPass final : public VPUIP::UnrollSwKernelBase<UnrollSwKernelPass> {
public:
    explicit UnrollSwKernelPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollSwKernelPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<SwKernelRewriter>(&ctx, _log);
    patterns.insert<ClusterSwKernelRewriter>(&ctx, _log);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createUnrollSwKernelPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUnrollSwKernelPass(Logger log) {
    return std::make_unique<UnrollSwKernelPass>(log);
}
