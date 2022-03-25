//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// Generated
//

#include <vpux/compiler/conversion/rewriters/generated/convert_declarations_to_VPUIP.hpp.inc>

//
// ConvertDeclarations2VPUIPPass
//

class ConvertDeclarations2VPUIPPass final : public ConvertDeclarations2VPUIPBase<ConvertDeclarations2VPUIPPass> {
public:
    explicit ConvertDeclarations2VPUIPPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertDeclarations2VPUIPPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<mlir::async::AsyncDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalDialect<VPURT::VPURTDialect>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();
    target.addLegalOp<IERT::SubViewOp, IERT::ConcatViewOp>();
    target.addLegalOp<IERT::GenericReshapeOp, IERT::PermuteCastOp>();
    target.addLegalOp<IERT::QuantizeCastOp>();
    target.addLegalOp<VPUIP::SwKernelOp>();
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
// createConvertDeclarations2VPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertDeclarations2VPUIPPass(Logger log) {
    return std::make_unique<ConvertDeclarations2VPUIPPass>(log);
}
