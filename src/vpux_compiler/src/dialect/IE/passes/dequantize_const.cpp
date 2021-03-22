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
// DequantizeConstPass
//

class DequantizeConstPass final : public IE::DequantizeConstBase<DequantizeConstPass> {
public:
    explicit DequantizeConstPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    void runOnFunction() final;

public:
    class DequantizeConst;

private:
    void passBody();

private:
    Logger _log;
};

void DequantizeConstPass::runOnFunction() {
    try {
        _log.trace("Run on Function '@{0}'", getFunction().sym_name());

        passBody();
    } catch (const std::exception& e) {
        (void)errorAt(getFunction(), "{0} Pass failed : {1}", getName(), e.what());
        signalPassFailure();
    }
}

//
// DequantizeConst
//

class DequantizeConstPass::DequantizeConst final : public mlir::OpRewritePattern<mlir::quant::DequantizeCastOp> {
public:
    DequantizeConst(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::quant::DequantizeCastOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::quant::DequantizeCastOp dCastOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DequantizeConstPass::DequantizeConst::matchAndRewrite(mlir::quant::DequantizeCastOp dCastOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    auto inputConst = dCastOp.arg().getDefiningOp<ConstantInterface>();

    if (inputConst == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got DequantizeCast Operation '{0}' with Constant input '{1}'", dCastOp->getLoc(), inputConst.getLoc());

    const auto qType = inputConst.getActualType();
    const auto qElemType = qType.getElementType().cast<mlir::quant::QuantizedType>();

    const auto newConstType =
            mlir::RankedTensorType::getChecked(dCastOp.getLoc(), qType.getShape(), qElemType.getExpressedType());
    if (newConstType == nullptr) {
        return mlir::failure();
    }

    const auto constAttr = dequantize(inputConst.getContent(), qType, dCastOp.getLoc());
    if (constAttr == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::ConstantOp>(dCastOp, newConstType, constAttr);
    return mlir::success();
}

//
// passBody
//

void DequantizeConstPass::passBody() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<DequantizeConst>(&ctx, _log.nest());

    auto func = getFunction();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDequantizeConstPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDequantizeConstPass(Logger log) {
    return std::make_unique<DequantizeConstPass>(log);
}
