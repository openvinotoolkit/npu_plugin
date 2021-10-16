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

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/Value.h>

using namespace vpux;

namespace {

//
// GenericConverter
//

template <class ConcreteOp>
class GenericConverter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    GenericConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx, benefitLow), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult GenericConverter<ConcreteOp>::matchAndRewrite(ConcreteOp originOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    const auto broadcastType =
            vpux::IE::AutoBroadcastTypeAttr::get(this->getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT);
    rewriter.replaceOpWithNewOp<IE::AndOp>(originOp, originOp.getType(), originOp.input(), originOp.input(),
                                           broadcastType, nullptr);

    return mlir::success();
}

//
// ConvertQuantizeOpsToEltwisePass
//

class ConvertQuantizeOpsToEltwisePass final :
        public IE::ConvertQuantizeOpsToEltwiseBase<ConvertQuantizeOpsToEltwisePass> {
public:
    explicit ConvertQuantizeOpsToEltwisePass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void ConvertQuantizeOpsToEltwisePass::safeRunOnFunc() {
    auto& ctx = getContext();

    // HW Eltwise supports only per-tensor bias/scale parameters
    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::QuantizeOp>([&](IE::QuantizeOp quantizeOp) {
        auto outElemType = quantizeOp.output().getType().cast<mlir::ShapedType>().getElementType();
        return outElemType.isa<mlir::quant::UniformQuantizedPerAxisType>();
    });
    target.addDynamicallyLegalOp<IE::DequantizeOp>([&](IE::DequantizeOp dequantizeOp) {
        auto inElemType = dequantizeOp.input().getType().cast<mlir::ShapedType>().getElementType();
        return inElemType.isa<mlir::quant::UniformQuantizedPerAxisType>();
    });
    target.addLegalOp<IE::AndOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<GenericConverter<IE::QuantizeOp>>(&ctx, _log);
    patterns.insert<GenericConverter<IE::DequantizeOp>>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertQuantizeOpsToEltwisePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertQuantizeOpsToEltwisePass(Logger log) {
    return std::make_unique<ConvertQuantizeOpsToEltwisePass>(log);
}
