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
// SwapTransposeWithFQ
//

class SwapTransposeWithFQ final : public IE::SwapTransposeWithFQBase<SwapTransposeWithFQ> {
public:
    explicit SwapTransposeWithFQ(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    class TransposeOpConverter;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

//
// TransposeOpConverter
//

class SwapTransposeWithFQ::TransposeOpConverter final : public mlir::OpRewritePattern<IE::TransposeOp> {
public:
    TransposeOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::TransposeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TransposeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwapTransposeWithFQ::TransposeOpConverter::matchAndRewrite(IE::TransposeOp origOp,
                                                                               mlir::PatternRewriter& rewriter) const {
    const auto transposeIn = origOp.input();
    VPUX_THROW_UNLESS(transposeIn != nullptr, "TransposeOpConverter: transpose input is a null pointer");
    auto origFqOp = transposeIn.getDefiningOp<IE::FakeQuantizeOp>();
    VPUX_THROW_UNLESS(origFqOp != nullptr, "TransposeOpConverter: transpose producer must be FQ");
    auto transposeOp =
            rewriter.create<IE::TransposeOp>(origOp->getLoc(), origFqOp.input(), nullptr, origOp.order_valueAttr());
    rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(
            origOp, transposeOp.output(), origFqOp.input_low(), origFqOp.input_high(), origFqOp.output_low(),
            origFqOp.output_high(), origFqOp.levels(), origFqOp.auto_broadcast());
    return mlir::success();
}

void SwapTransposeWithFQ::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto hasFQInput = [](IE::TransposeOp op) -> bool {
        const auto transposeIn = op.input();
        if (!transposeIn) {
            return true;
        }

        // FIXME check that FakeQuantize has per-tensor quantization
        auto maybeFqOp = transposeIn.getDefiningOp<IE::FakeQuantizeOp>();
        if (maybeFqOp == nullptr) {
            return true;
        }

        return (maybeFqOp.input().dyn_cast<mlir::BlockArgument>() == nullptr);
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::TransposeOp>(hasFQInput);
    target.addLegalOp<IE::FakeQuantizeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<SwapTransposeWithFQ::TransposeOpConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createSwapTransposeWithFQPass(Logger log) {
    return std::make_unique<SwapTransposeWithFQ>(log);
}
