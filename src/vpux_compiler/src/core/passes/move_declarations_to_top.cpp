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

#include "vpux/compiler/core/ops_interfaces.hpp"

#include "vpux/utils/core/range.hpp"

using namespace vpux;

namespace {

//
// MoveDeclarationsToTopPass
//

class MoveDeclarationsToTopPass final : public MoveDeclarationsToTopBase<MoveDeclarationsToTopPass> {
public:
    explicit MoveDeclarationsToTopPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void MoveDeclarationsToTopPass::safeRunOnFunc() {
    auto& block = getFunction().getBody().front();

    SmallVector<mlir::Operation*> allDeclOps;
    for (auto& op : block) {
        if (op.hasTrait<DeclarationOp>()) {
            allDeclOps.push_back(&op);
        }
    }

    if (allDeclOps.empty()) {
        return;
    }

    auto* firstDeclOp = allDeclOps.front();
    firstDeclOp->moveBefore(&block, block.begin());

    for (auto i : irange(allDeclOps.size() - 1)) {
        allDeclOps[i + 1]->moveAfter(allDeclOps[i]);
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::createMoveDeclarationsToTopPass(Logger log) {
    return std::make_unique<MoveDeclarationsToTopPass>(log);
}
