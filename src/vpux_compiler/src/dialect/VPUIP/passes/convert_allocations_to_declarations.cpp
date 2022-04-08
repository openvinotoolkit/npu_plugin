//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// Generated
//

#include <vpux/compiler/dialect/VPUIP/rewriters/generated/convert_allocations_to_declarations.hpp.inc>

//
// ConvertAllocationsToDeclarationsPass
//

class ConvertAllocationsToDeclarationsPass final :
        public VPUIP::ConvertAllocationsToDeclarationsBase<ConvertAllocationsToDeclarationsPass> {
public:
    explicit ConvertAllocationsToDeclarationsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertAllocationsToDeclarationsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<mlir::async::AsyncDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalDialect<VPURT::VPURTDialect>();
    target.addIllegalOp<VPUIP::StaticAllocOp>();
    target.addLegalOp<VPUIP::SwKernelOp>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();
    target.markOpRecursivelyLegal<VPUIP::SwKernelOp>([&](mlir::Operation*) {
        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    populateWithGenerated(patterns);

    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertAllocationsToDeclarationsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertAllocationsToDeclarationsPass(Logger log) {
    return std::make_unique<ConvertAllocationsToDeclarationsPass>(log);
}
