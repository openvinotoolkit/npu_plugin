//
// Copyright Intel Corporation.
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
    mlir::LogicalResult matchAndRewrite(mlir::async::ExecuteOp execOp, ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult InlineAsyncRegion::matchAndRewrite(mlir::async::ExecuteOp execOp, ArrayRef<mlir::Value> operands,
                                                       mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found 'async.execute' operation at '{0}'", execOp->getLoc());

    auto yieldOp = mlir::dyn_cast_or_null<mlir::async::YieldOp>(execOp.getBody()->getTerminator());
    VPUX_THROW_UNLESS(yieldOp != nullptr, "'async.execute' body doesn't have corresponding terminator");

    auto barrierOp = rewriter.create<VPUIP::DeclareVirtualBarrierOp>(execOp.getLoc());

    mlir::async::ExecuteOp::Adaptor newArgs(operands, execOp->getAttrDictionary());

    _log.nest(1).trace("Traverse 'async.execute' body");
    for (auto& op : execOp.getBody()->without_terminator()) {
        if (auto taskOp = mlir::dyn_cast<VPUIP::TaskOpInterface>(op)) {
            _log.nest(2).trace("Found VPUIP task operation '{0}' as '{1}'", op.getName(), op.getLoc());

            if (!newArgs.dependencies().empty()) {
                _log.nest(3).trace("Append it's wait barrier list");
                taskOp.waitBarriersMutable().append(newArgs.dependencies());
            }

            if (!execOp.token().use_empty()) {
                _log.nest(3).trace("Append it's update barrier list");
                taskOp.updateBarriersMutable().append(barrierOp.barrier());
            }
        }
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
    mlir::LogicalResult matchAndRewrite(mlir::async::AwaitOp waitOp, ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult RemoveWait::matchAndRewrite(mlir::async::AwaitOp waitOp, ArrayRef<mlir::Value> operands,
                                                mlir::ConversionPatternRewriter& rewriter) const {
    VPUX_THROW_UNLESS(waitOp.result() != nullptr, "'async.await' Operation without result is not supported");
    VPUX_THROW_UNLESS(waitOp.result().hasOneUse(), "'async.await' has more than one consumer");
    VPUX_THROW_UNLESS(mlir::isa<mlir::ReturnOp>(*waitOp.result().user_begin()),
                      "'async.await' has non 'return' consumer");

    rewriter.replaceOp(waitOp, operands[0]);
    return mlir::success();
}

//
// safeRunOnFunc
//

template <class ConcreteType>
mlir::Value dummyConverter(mlir::OpBuilder& builder, ConcreteType type, mlir::ValueRange inputs, mlir::Location loc) {
    SmallVector<mlir::Value> results;
    builder.createOrFold<mlir::UnrealizedConversionCastOp>(results, loc, type, inputs);
    return results.front();
}

void ConvertAsyncOps2VPUIPPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::TypeConverter typeConverter;

    typeConverter.addConversion([](mlir::async::TokenType token) {
        return VPUIP::BarrierType::get(token.getContext());
    });
    typeConverter.addTargetMaterialization(dummyConverter<VPUIP::BarrierType>);
    typeConverter.addArgumentMaterialization(dummyConverter<VPUIP::BarrierType>);
    typeConverter.addSourceMaterialization(dummyConverter<mlir::async::TokenType>);

    typeConverter.addConversion([](mlir::async::ValueType future) {
        return future.getValueType();
    });
    typeConverter.addTargetMaterialization(dummyConverter<mlir::MemRefType>);
    typeConverter.addArgumentMaterialization(dummyConverter<mlir::MemRefType>);
    typeConverter.addSourceMaterialization(dummyConverter<mlir::async::ValueType>);

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();

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
