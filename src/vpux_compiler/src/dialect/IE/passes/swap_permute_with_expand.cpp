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

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/TypeSwitch.h>
#include <vpux/compiler/conversion.hpp>

using namespace vpux;

namespace {

//
// SwapPermuteWithExpand
//

class SwapPermuteWithExpand final : public IE::SwapPermuteWithExpandBase<SwapPermuteWithExpand> {
public:
    explicit SwapPermuteWithExpand(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    class MemPermuteOpConverter;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

//
// MemPermuteOpConverter
//

class SwapPermuteWithExpand::MemPermuteOpConverter final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    MemPermuteOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwapPermuteWithExpand::MemPermuteOpConverter::matchAndRewrite(
        IE::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto permuteIn = origOp.input();
    VPUX_THROW_UNLESS(permuteIn != nullptr, "MemPermuteOpConverter: permute input is a null pointer");
    auto origExpandOp = permuteIn.getDefiningOp<IE::ExpandOp>();
    VPUX_THROW_UNLESS(origExpandOp != nullptr, "MemPermuteOpConverter: permute producer must be ExpandOp");
    auto permuteOp = rewriter.create<IE::MemPermuteOp>(origOp->getLoc(), origExpandOp.input(), origOp.dst_orderAttr(),
                                                       origOp.mem_permAttr());
    rewriter.replaceOpWithNewOp<IE::ExpandOp>(origOp, permuteOp.output(), origExpandOp.pads_beginAttr(),
                                              origExpandOp.pads_endAttr());
    return mlir::success();
}

void SwapPermuteWithExpand::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto hasExpandInput = [](IE::MemPermuteOp op) -> bool {
        const auto permuteIn = op.input();
        if (!permuteIn) {
            return true;
        }
        const auto maybeExpandOp = permuteIn.getDefiningOp<IE::ExpandOp>();
        return maybeExpandOp == nullptr;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::MemPermuteOp>(hasExpandInput);
    target.addLegalOp<IE::ExpandOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<SwapPermuteWithExpand::MemPermuteOpConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createSwapPermuteWithExpandPass(Logger log) {
    return std::make_unique<SwapPermuteWithExpand>(log);
}
