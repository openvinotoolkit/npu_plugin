//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/pad_extract.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/TypeSwitch.h>
#include <vpux/compiler/conversion.hpp>

using namespace vpux;

namespace {

//
// SwapWithTranspose
//

class SwapWithTranspose final : public mlir::OpRewritePattern<IE::TransposeOp> {
public:
    SwapWithTranspose(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::TransposeOp>(ctx), _log(log) {
        this->setDebugName("SwapPadLayer::SwapWithTranspose");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::TransposeOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwapWithTranspose::matchAndRewrite(IE::TransposeOp originOp,
                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", originOp->getName(), originOp->getLoc());
    if (!originOp.input().hasOneUse()) {
        return matchFailed(rewriter, originOp, "Operation {0} is not the only user of its operand",
                           originOp->getName());
    }

    auto padLayer = originOp.input().getDefiningOp<IE::PadOp>();
    if (padLayer == nullptr) {
        return matchFailed(rewriter, originOp, "Producer is not a Pad operation");
    }

    auto newTranspose =
            rewriter.create<IE::TransposeOp>(originOp.getLoc(), padLayer.input(), nullptr, originOp.order_valueAttr());

    const auto orderAttr = originOp.order_valueAttr();
    const auto order = DimsOrder::fromAffineMap(orderAttr.getValue());

    const auto permutation = order.toPermutation();

    auto padsBegin = vpux::IE::extractPads(padLayer.pads_begin_attrAttr(), _log);
    if (mlir::failed(padsBegin)) {
        return mlir::failure();
    }

    auto padsEnd = vpux::IE::extractPads(padLayer.pads_end_attrAttr(), _log);
    if (mlir::failed(padsEnd)) {
        return mlir::failure();
    }

    const auto padsBeginValue = padsBegin.getValue();
    const auto padsEndValue = padsEnd.getValue();

    VPUX_THROW_UNLESS(permutation.size() == padsBeginValue.size() && padsBeginValue.size() == padsEndValue.size(),
                      "Permutation size {0} and pads {1} don't match", permutation.size(), padsBeginValue.size());

    SmallVector<int64_t> beginTransposed(permutation.size(), 0);
    SmallVector<int64_t> endTransposed(permutation.size(), 0);

    for (auto p : permutation | indexed) {
        beginTransposed[p.index()] = padsBeginValue[p.value().ind()];
        endTransposed[p.index()] = padsEndValue[p.value().ind()];
    }

    rewriter.replaceOpWithNewOp<IE::PadOp>(originOp, newTranspose, nullptr, nullptr, nullptr,
                                           getIntArrayAttr(originOp.getContext(), makeArrayRef(beginTransposed)),
                                           getIntArrayAttr(originOp.getContext(), makeArrayRef(endTransposed)),
                                           padLayer.pad_value_attrAttr(), padLayer.mode());

    return mlir::success();
}

//
// SwapPadLayer
//

class SwapPadLayer final : public IE::SwapPadLayerBase<SwapPadLayer> {
public:
    explicit SwapPadLayer(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void SwapPadLayer::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SwapWithTranspose>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createSwapPadLayerPass(Logger log) {
    return std::make_unique<SwapPadLayer>(log);
}
