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

#include "vpux/compiler/core/passes.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/StandardOps/Transforms/ComposeSubView.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// ComposeSubViewPass
//

class ComposeSubViewPass final : public ComposeSubViewBase<ComposeSubViewPass> {
public:
    explicit ComposeSubViewPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ComposeSubViewPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    mlir::OwningRewritePatternList patterns(&ctx);
    mlir::populateComposeSubViewPatterns(patterns, &ctx);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::createComposeSubViewPass(Logger log) {
    return std::make_unique<ComposeSubViewPass>(log);
}
