//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/StandardOps/Transforms/FuncConversions.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// BufferizeFuncAndReturn
//

class BufferizeFuncAndReturn final : public BufferizeFuncAndReturnBase<BufferizeFuncAndReturn> {
public:
    explicit BufferizeFuncAndReturn(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void BufferizeFuncAndReturn::safeRunOnModule() {
    auto& ctx = getContext();

    vpux::BufferizeTypeConverter typeConverter;

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<mlir::ModuleOp>();
    target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp op) {
        return typeConverter.isSignatureLegal(op.getType()) && typeConverter.isLegal(&op.getBody());
    });
    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        return mlir::isNotBranchOpInterfaceOrReturnLikeOp(op) ||
               mlir::isLegalForReturnOpTypeConversionPattern(op, typeConverter);
    });
    vpux::populateBufferizeMaterializationLegality(target);

    mlir::RewritePatternSet patterns(&ctx);
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::FuncOp>(patterns, typeConverter);
    mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);

    auto module = getOperation();
    if (mlir::failed(applyFullConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createBufferizeFuncAndReturn
//

std::unique_ptr<mlir::Pass> vpux::createBufferizeFuncAndReturnPass(Logger log) {
    return std::make_unique<BufferizeFuncAndReturn>(log);
}
