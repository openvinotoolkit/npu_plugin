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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

SmallVector<uint32_t> getOrder(DimsOrder inOrder, DimsOrder outOrder) {
    auto inPerm = inOrder.toPermutation();
    auto outPerm = outOrder.toPermutation();
    SmallVector<uint32_t> memPerm(inPerm.size());
    for (auto p : outPerm | indexed) {
        memPerm[p.index()] = static_cast<uint32_t>(inOrder.dimPos(p.value()));
    }
    return memPerm;
}

//
// ConvertToMemPermutePass
//

class ConvertToMemPermutePass final : public IE::ConvertToMemPermuteBase<ConvertToMemPermutePass> {
public:
    explicit ConvertToMemPermutePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class ReorderOpConverter;
    class TransposeOpConverter;

private:
    void safeRunOnFunc() final;
};

//
// ReorderOpConverter
//

class ConvertToMemPermutePass::ReorderOpConverter final : public mlir::OpRewritePattern<IE::ReorderOp> {
public:
    ReorderOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ReorderOp>(ctx), _log(log) {
        setDebugName("ConvertToMemPermutePass::ReorderOpConverter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ReorderOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertToMemPermutePass::ReorderOpConverter::matchAndRewrite(
        IE::ReorderOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::Reorder Operation '{0}'", origOp->getLoc());

    auto inOrder = DimsOrder::fromValue(origOp.input());
    auto outOrder = DimsOrder::fromValue(origOp.output());

    auto memPerm = mlir::AffineMap::getPermutationMap(makeArrayRef(getOrder(inOrder, outOrder)), origOp->getContext());
    auto memPermAttr = mlir::AffineMapAttr::get(memPerm);

    rewriter.replaceOpWithNewOp<IE::MemPermuteOp>(origOp, origOp.input(), origOp.dstOrderAttr(), memPermAttr);

    _log.trace("Replaced with 'IE::MemPermute'");

    return mlir::success();
}

//
// TransposeOpConverter
//

class ConvertToMemPermutePass::TransposeOpConverter final : public mlir::OpRewritePattern<IE::TransposeOp> {
public:
    TransposeOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::TransposeOp>(ctx), _log(log) {
        setDebugName("ConvertToMemPermutePass::TransposeOpConverter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TransposeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertToMemPermutePass::TransposeOpConverter::matchAndRewrite(
        IE::TransposeOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::Transpose Operation '{0}'", origOp->getLoc());
    VPUX_THROW_UNLESS(origOp.order_value().hasValue(), "IE::Transpose Operation doesn't have order_value attribute");

    auto outputOrder = DimsOrder::fromValue(origOp.output());
    auto dstOrder = mlir::AffineMapAttr::get(outputOrder.toPermutationAffineMap(origOp.getContext()));
    auto inputOrder = DimsOrder::fromValue(origOp.input());
    auto inPerm = inputOrder.toPermutationAffineMap(origOp.getContext());
    auto memPerm = inPerm.compose(origOp.order_value().getValue());
    auto memPermAttr = mlir::AffineMapAttr::get(memPerm);

    rewriter.replaceOpWithNewOp<IE::MemPermuteOp>(origOp, origOp.input(), dstOrder, memPermAttr);

    _log.trace("Replaced with 'IE::MemPermute'");

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertToMemPermutePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::TransposeOp>();
    target.addIllegalOp<IE::ReorderOp>();
    target.addLegalOp<IE::MemPermuteOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<TransposeOpConverter>(&ctx, _log);
    patterns.insert<ReorderOpConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertToMemPermutePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertToMemPermutePass(Logger log) {
    return std::make_unique<ConvertToMemPermutePass>(log);
}
