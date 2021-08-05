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
#include "vpux/compiler/dialect/VPUIP/attributes/arch.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// TimestampRewrite
//

class TimestampRewrite final : public mlir::OpRewritePattern<IERT::TimestampOp> {
public:
    TimestampRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IERT::TimestampOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::TimestampOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult TimestampRewrite::matchAndRewrite(IERT::TimestampOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Timestamp Operation '{0}'", origOp->getLoc());

    auto origType = origOp.getType();
    VPUX_THROW_UNLESS(origType.getNumElements() == 1, "Got wrong elements number for TimestampOp");
    VPUX_THROW_UNLESS(origType.getElementType() == getUInt32Type(getContext()),
                      "Got wrong element type for TimestampOp");

    auto timerType =
            mlir::MemRefType::get(origType.getShape(), origType.getElementType(), {},
                                  VPUIP::MemoryLocationAttr::get(getContext(), VPUIP::MemoryLocation::AbsoluteAddr));

    rewriter.replaceOpWithNewOp<VPUIP::DeclareTensorOp>(origOp, timerType, VPUIP::MemoryLocation::AbsoluteAddr, 0,
                                                        VPUIP::HW_TIMER_ABSOLUTE_ADDR);
    _log.trace("Replaced with 'VPUIP.DeclareTensorOp'");

    return mlir::success();
}  // namespace

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
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();
    target.addLegalOp<IERT::GenericReshapeOp, IERT::ConcatViewOp, mlir::memref::SubViewOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<TimestampRewrite>(&ctx, _log);
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
