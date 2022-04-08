//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// ConvertAsyncOps2VPUIPPass
//

class ConvertAsyncOps2VPUIPPass final : public ConvertAsyncOps2VPUIPBase<ConvertAsyncOps2VPUIPPass> {
public:
    explicit ConvertAsyncOps2VPUIPPass(Logger log) {
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
    InlineAsyncRegion(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<mlir::async::ExecuteOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::async::ExecuteOp execOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
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

void ConvertAsyncOps2VPUIPPass::safeRunOnFunc() {
    auto& ctx = getContext();

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

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<InlineAsyncRegion>(typeConverter, &ctx, _log);
    patterns.insert<RemoveWait>(typeConverter, &ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertAsyncOps2VPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertAsyncOps2VPUIPPass(Logger log) {
    return std::make_unique<ConvertAsyncOps2VPUIPPass>(log);
}
