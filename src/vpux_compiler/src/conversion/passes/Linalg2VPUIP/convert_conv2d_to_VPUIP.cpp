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

#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>

using namespace vpux;

namespace {

//
// ConvertConv2D
//

class ConvertConv2D final : public mlir::OpConversionPattern<mlir::linalg::Conv2DOp> {
public:
    ConvertConv2D(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<mlir::linalg::Conv2DOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::linalg::Conv2DOp origOp, OpAdaptor adaptor, 
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertConv2D::matchAndRewrite(mlir::linalg::Conv2DOp origOp,
                                                   OpAdaptor adaptor,
                                                   mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Convolution Operation '{0}'", origOp->getLoc());

    (void)adaptor;
    (void)rewriter;
    return mlir::success();
}

//
// ConvertConv2D2VPUIPPass
//

class ConvertConv2D2VPUIPPass final : public ConvertConv2D2VPUIPBase<ConvertConv2D2VPUIPPass> {
public:
    explicit ConvertConv2D2VPUIPPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertConv2D2VPUIPPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalDialect<mlir::linalg::LinalgDialect>();
    target.addLegalDialect<VPUIP::VPUIPDialect>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ConvertConv2D>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertConv2D2VPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertConv2D2VPUIPPass(Logger log) {
    return std::make_unique<ConvertConv2D2VPUIPPass>(log);
}
