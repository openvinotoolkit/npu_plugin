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
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/linear_scan.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/DenseSet.h>

using namespace vpux;

namespace {

class LinearizationPass final : public IERT::LinearizationBase<LinearizationPass> {
public:
    explicit LinearizationPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void LinearizationPass::safeRunOnModule() {
    auto module = getOperation();

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    auto& depsInfo = getChildAnalysis<AsyncDepsInfo>(netFunc);

    mlir::async::ExecuteOp prevExecOp;
    for (auto curExecOp : netFunc.getOps<mlir::async::ExecuteOp>()) {
        if (prevExecOp != nullptr) {
            _log.trace("Add explicit dependency from '{0}' to '{1}'", prevExecOp->getLoc(), curExecOp->getLoc());
            depsInfo.addDependency(prevExecOp, curExecOp);
        }

        prevExecOp = curExecOp;
    }

    depsInfo.updateTokenDependencies();
}

}  // namespace

//
// createLinearizationPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createLinearizationPass(Logger log) {
    return std::make_unique<LinearizationPass>(log);
}