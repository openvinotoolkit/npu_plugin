//
// Copyright Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/utils/quantization.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// QuantizeConstPass
//

class QuantizeConstPass final : public IE::QuantizeConstBase<QuantizeConstPass> {
public:
    explicit QuantizeConstPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    void runOnFunction() final;

public:
    class QuantizeConst;

private:
    void passBody();

private:
    Logger _log;
};

void QuantizeConstPass::runOnFunction() {
    try {
        _log.trace("Run on Function '@{0}'", getFunction().sym_name());

        passBody();
    } catch (const std::exception& e) {
        (void)errorAt(getFunction(), "{0} Pass failed : {1}", getName(), e.what());
        signalPassFailure();
    }
}

//
// QuantizeConst
//

class QuantizeConstPass::QuantizeConst final : public mlir::OpRewritePattern<mlir::quant::QuantizeCastOp> {
public:
    QuantizeConst(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::quant::QuantizeCastOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::quant::QuantizeCastOp qCastOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult QuantizeConstPass::QuantizeConst::matchAndRewrite(mlir::quant::QuantizeCastOp qCastOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    auto inputConst = qCastOp.arg().getDefiningOp<ConstantInterface>();

    if (inputConst == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got QuantizeCast Operation '{0}' with Constant input '{1}'", qCastOp->getLoc(), inputConst.getLoc());

    const auto qType = qCastOp.getType().cast<mlir::ShapedType>();

    const auto constAttr = quantize(inputConst.getContent(), qType, qCastOp.getLoc());
    if (constAttr == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::ConstantOp>(qCastOp, qType, constAttr);
    return mlir::success();
}

//
// passBody
//

void QuantizeConstPass::passBody() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<QuantizeConst>(&ctx, _log.nest());

    auto func = getFunction();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createQuantizeConstPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createQuantizeConstPass(Logger log) {
    return std::make_unique<QuantizeConstPass>(log);
}
