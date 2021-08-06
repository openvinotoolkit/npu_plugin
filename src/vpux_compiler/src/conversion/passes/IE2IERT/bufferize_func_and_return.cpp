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
    mlir::populateFuncOpTypeConversionPattern(patterns, typeConverter);
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
