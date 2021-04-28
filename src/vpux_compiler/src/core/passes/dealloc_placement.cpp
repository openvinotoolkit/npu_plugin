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
