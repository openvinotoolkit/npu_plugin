//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/cost_model.hpp"

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// ConvertAsyncOpsToTasksPass
//

class ConvertAsyncOpsToTasksPass final : public VPUIP::ConvertAsyncOpsToTasksBase<ConvertAsyncOpsToTasksPass> {
public:
    explicit ConvertAsyncOpsToTasksPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// InlineAsyncRegion
//

class InlineAsyncRegion final : public mlir::OpConversionPattern<mlir::async::ExecuteOp> {
public:
    InlineAsyncRegion(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log, VPU::ArchKind arch,
                      std::shared_ptr<VPUNN::VPUCostModel> costModel)
            : mlir::OpConversionPattern<mlir::async::ExecuteOp>(typeConverter, ctx),
              _log(log),
              _arch(arch),
              _costModel(costModel) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::async::ExecuteOp execOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
    VPU::ArchKind _arch;
    std::shared_ptr<VPUNN::VPUCostModel> _costModel;
};

mlir::LogicalResult InlineAsyncRegion::matchAndRewrite(mlir::async::ExecuteOp execOp, OpAdaptor newArgs,
                                                       mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found 'async.execute' operation at '{0}'", execOp->getLoc());

    auto yieldOp = mlir::dyn_cast_or_null<mlir::async::YieldOp>(execOp.getBody()->getTerminator());
    VPUX_THROW_UNLESS(yieldOp != nullptr, "'async.execute' body doesn't have corresponding terminator");

    auto barrierOp = rewriter.create<VPURT::DeclareVirtualBarrierOp>(execOp.getLoc());

    _log.nest(1).trace("Traverse 'async.execute' body");

    SmallVector<mlir::Operation*> ops;
    for (auto& op : execOp.getBody()->without_terminator()) {
        if (!mlir::isa<VPURT::DeclareBufferOp>(op)) {
            ops.push_back(&op);
        }
    }

    bool hasCycleCost = execOp->hasAttr(cycleBegin) && execOp->hasAttr(cycleEnd);
    bool singleNestedOp = ops.size() == 1;
    size_t currentStart = 0;
    if (hasCycleCost) {
        currentStart = checked_cast<size_t>(execOp->getAttrOfType<mlir::IntegerAttr>(cycleBegin).getInt());
    }

    for (auto op : ops) {
        mlir::SmallVector<mlir::Value> waitBarriers, updateBarriers;
        if (!newArgs.dependencies().empty()) {
            _log.nest(3).trace("Append it's wait barrier list");
            waitBarriers.append(newArgs.dependencies().begin(), newArgs.dependencies().end());
        }
        if (!execOp.token().use_empty()) {
            _log.nest(3).trace("Append it's update barrier list");
            updateBarriers.push_back(barrierOp.barrier());
        }

        rewriter.setInsertionPoint(op);
        auto taskOp = rewriter.create<VPURT::TaskOp>(op->getLoc(), waitBarriers, updateBarriers);
        auto& block = taskOp.body().emplaceBlock();
        op->moveBefore(&block, block.end());

        if (!hasCycleCost)
            continue;

        if (singleNestedOp) {
            taskOp->setAttr(cycleBegin, execOp->getAttr(cycleBegin));
            taskOp->setAttr(cycleEnd, execOp->getAttr(cycleEnd));
        } else {
            auto tilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(op);
            auto dmaOp = (tilingOp == nullptr) ? mlir::dyn_cast<VPUIP::NNDMAOp>(op)
                                               : mlir::dyn_cast<VPUIP::NNDMAOp>(tilingOp.getInnerTaskOp());
            if (dmaOp != nullptr) {
                size_t dmaCost = getDMACost(dmaOp.input(), dmaOp.output(), _arch, _costModel);
                taskOp->setAttr(cycleBegin, getIntAttr(taskOp->getContext(), currentStart));
                taskOp->setAttr(cycleCostAttrName, getIntAttr(taskOp->getContext(), dmaCost));
                taskOp->setAttr(cycleEnd, getIntAttr(taskOp->getContext(), currentStart + dmaCost));
                currentStart += dmaCost;
            } else {
                VPUX_THROW("Found unsupported operation nested inside async.execute: {0}", op);
            }
        }
    }
    if (!singleNestedOp && hasCycleCost) {
        size_t execOpCycleEnd = checked_cast<size_t>(execOp->getAttrOfType<mlir::IntegerAttr>(cycleEnd).getInt());
        VPUX_THROW_UNLESS(currentStart == execOpCycleEnd,
                          "Calculated operation cost doesn't add up to expected: {0} != {1}", currentStart,
                          execOpCycleEnd);
    }

    SmallVector<mlir::Value> newResults;
    newResults.reserve(execOp->getNumResults());
    newResults.push_back(barrierOp.barrier());
    newResults.append(yieldOp.operands().begin(), yieldOp.operands().end());

    rewriter.eraseOp(yieldOp);
    rewriter.mergeBlockBefore(execOp.getBody(), execOp, newArgs.operands());
    rewriter.replaceOp(execOp, newResults);

    return mlir::success();
}

//
// RemoveWait
//

class RemoveWait final : public mlir::OpConversionPattern<mlir::async::AwaitOp> {
public:
    RemoveWait(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<mlir::async::AwaitOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::async::AwaitOp waitOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult RemoveWait::matchAndRewrite(mlir::async::AwaitOp waitOp, OpAdaptor newArgs,
                                                mlir::ConversionPatternRewriter& rewriter) const {
    VPUX_THROW_UNLESS(waitOp.result() != nullptr, "'async.await' Operation without result is not supported");

    // Pure view like operation were replaced with DeclareTensorOp
    // so the remaining Await ops have no users
    if (waitOp.result().use_empty()) {
        rewriter.eraseOp(waitOp);
        return mlir::success();
    }

    // If you faced with "'async.await' has more than one consumer" make sure you add your layer
    // into src/vpux_compiler/src/dialect/VPUIP/ops.cpp `redirectOpInterfacesForIE(...)`
    // and `redirectOpInterfacesForIERT(...)`
    VPUX_THROW_UNLESS(waitOp.result().hasOneUse(), "'async.await' doesn't have only one consumer");
    VPUX_THROW_UNLESS(mlir::isa<mlir::ReturnOp>(*waitOp.result().user_begin()),
                      "'async.await' has non 'return' consumer");

    rewriter.replaceOp(waitOp, newArgs.operand());
    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertAsyncOpsToTasksPass::safeRunOnFunc() {
    auto& ctx = getContext();

    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    mlir::TypeConverter typeConverter;

    typeConverter.addConversion([](mlir::async::TokenType token) {
        return VPURT::BarrierType::get(token.getContext());
    });
    typeConverter.addTargetMaterialization(dummyConverter<VPURT::BarrierType>);
    typeConverter.addArgumentMaterialization(dummyConverter<VPURT::BarrierType>);
    typeConverter.addSourceMaterialization(dummyConverter<mlir::async::TokenType>);

    typeConverter.addConversion([](mlir::async::ValueType future) {
        return future.getValueType();
    });
    typeConverter.addTargetMaterialization(dummyConverter<mlir::MemRefType>);
    typeConverter.addArgumentMaterialization(dummyConverter<mlir::MemRefType>);
    typeConverter.addSourceMaterialization(dummyConverter<mlir::async::ValueType>);

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalDialect<VPURT::VPURTDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();

    target.addLegalOp<VPUIP::SwKernelOp>();
    target.markOpRecursivelyLegal<VPUIP::SwKernelOp>([&](mlir::Operation*) {
        return true;
    });

    const auto arch = VPU::getArch(module);
    const auto costModel = VPU::createCostModel(arch);
    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<InlineAsyncRegion>(typeConverter, &ctx, _log, arch, costModel);
    patterns.insert<RemoveWait>(typeConverter, &ctx, _log);

    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertAsyncOpsToTasksPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertAsyncOpsToTasksPass(Logger log) {
    return std::make_unique<ConvertAsyncOpsToTasksPass>(log);
}
