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
// MergeFakeQuantPass
//

class MergeFakeQuantPass final : public IE::MergeFakeQuantBase<MergeFakeQuantPass> {
public:
    explicit MergeFakeQuantPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class UseFakeQuant;

private:
    void safeRunOnFunc() final;
};

//
// UseFakeQuant
//

class MergeFakeQuantPass::UseFakeQuant final : public mlir::OpRewritePattern<mlir::quant::DequantizeCastOp> {
public:
    UseFakeQuant(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::quant::DequantizeCastOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::quant::DequantizeCastOp dCastOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MergeFakeQuantPass::UseFakeQuant::matchAndRewrite(mlir::quant::DequantizeCastOp dCastOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    auto qCastOp = dCastOp.arg().getDefiningOp<mlir::quant::QuantizeCastOp>();

    if (qCastOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got QuantizeCast ('{0}') -> DequantizeCast ('{1}') pair", qCastOp.getLoc(), dCastOp.getLoc());

    const auto qType = dCastOp.arg().getType().cast<mlir::ShapedType>();

    uint32_t levels = 0;
    mlir::RankedTensorType attrType;
    mlir::DenseElementsAttr rMinAttr, rMaxAttr;

    if (mlir::failed(getFakeQuantParams(qType, levels, attrType, rMinAttr, rMaxAttr, dCastOp.getLoc()))) {
        return mlir::failure();
    }

    auto rMinOp = rewriter.create<IE::ConstantOp>(dCastOp.getLoc(), attrType, rMinAttr);
    auto rMaxOp = rewriter.create<IE::ConstantOp>(dCastOp.getLoc(), attrType, rMaxAttr);

    rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(dCastOp, qCastOp.arg(), rMinOp.output(), rMaxOp.output(),
                                                    rMinOp.output(), rMaxOp.output(), levels,
                                                    IE::AutoBroadcastType::NUMPY);

    return mlir::success();
}

//
// safeRunOnFunc
//

void MergeFakeQuantPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<UseFakeQuant>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMergeFakeQuantPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createMergeFakeQuantPass(Logger log) {
    return std::make_unique<MergeFakeQuantPass>(log);
}
