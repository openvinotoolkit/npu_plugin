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

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

namespace {

//
// DeallocPlacementPass
//

class DeallocPlacementPass final : public DeallocPlacementBase<DeallocPlacementPass> {
public:
    explicit DeallocPlacementPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    static mlir::Operation* getLastUser(mlir::Value val, const AliasesInfo& info);
};

//
// getLastUse
//

mlir::Operation* DeallocPlacementPass::getLastUser(mlir::Value val, const AliasesInfo& info) {
    auto* producer = val.getDefiningOp();
    VPUX_THROW_UNLESS(producer != nullptr && mlir::isa<mlir::memref::AllocOp>(producer),
                      "Wrong allocated value producer");

    auto* lastUser = producer;

    const auto& aliases = info.getAliases(val);

    for (auto alias : aliases) {
        VPUX_THROW_UNLESS(alias.getParentBlock() == val.getParentBlock(),
                          "Alias '{0}' doesn't belong to the same block as '{1}'", alias, val);

        for (auto* user : alias.getUsers()) {
            VPUX_THROW_UNLESS(user != nullptr && !mlir::isa<mlir::memref::DeallocOp>(user),
                              "Wrong allocated value user");

            if (lastUser->isBeforeInBlock(user)) {
                lastUser = user;
            }
        }
    }

    return lastUser;
}

//
// safeRunOnFunc
//

void DeallocPlacementPass::safeRunOnFunc() {
    auto& aliasInfo = getAnalysis<AliasesInfo>();

    auto func = getFunction();

    func.walk([&](mlir::memref::AllocOp alloc) {
        auto* lastUser = getLastUser(alloc.memref(), aliasInfo);

        auto* nextOp = lastUser->getNextNode();
        VPUX_THROW_UNLESS(nextOp != nullptr, "Missing next operation after '{0}'", lastUser->getLoc());

        OpBuilderLogger builderLogger(_log);
        mlir::OpBuilder builder(nextOp, &builderLogger);
        builder.create<mlir::memref::DeallocOp>(alloc.getLoc(), alloc.memref());
    });
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::createDeallocPlacementPass(Logger log) {
    return std::make_unique<DeallocPlacementPass>(log);
}
