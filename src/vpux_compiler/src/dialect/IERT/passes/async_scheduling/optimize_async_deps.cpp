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

#include "vpux/compiler/dialect/IERT/passes.hpp"

#include "vpux/compiler/core/async_deps_info.hpp"

using namespace vpux;

namespace {

//
// OptimizeAsyncDepsPass
//

class OptimizeAsyncDepsPass final : public IERT::OptimizeAsyncDepsBase<OptimizeAsyncDepsPass> {
public:
    explicit OptimizeAsyncDepsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeAsyncDepsPass::safeRunOnFunc() {
    auto& depsInfo = getAnalysis<AsyncDepsInfo>();
    depsInfo.optimizeDepsMap();
    depsInfo.updateTokenDependencies();
}

}  // namespace

//
// createOptimizeAsyncDepsPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createOptimizeAsyncDepsPass(Logger log) {
    return std::make_unique<OptimizeAsyncDepsPass>(log);
}
