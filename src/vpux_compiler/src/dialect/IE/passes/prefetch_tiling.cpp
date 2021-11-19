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


using namespace vpux;

namespace {

//
// IsolatedTilingPass
//
class PrefetchTilingPass final : public IE::PrefetchTilingBase<PrefetchTilingPass> {
public:
    explicit PrefetchTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void PrefetchTilingPass::safeRunOnFunc() {
//    auto& ctx = getContext();
}


} // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPrefetchTilingPass(Logger log) {
    return std::make_unique<PrefetchTilingPass>(log);
}
